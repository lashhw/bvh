#ifndef BVH_NODE_INTERSECTORS_HPP
#define BVH_NODE_INTERSECTORS_HPP

#include <cmath>
#include <mpfr.h>

#include "bvh/vector.hpp"
#include "bvh/ray.hpp"
#include "bvh/platform.hpp"
#include "bvh/utilities.hpp"

namespace bvh {

/// Base class for ray-node intersection algorithms. Does ray octant classification.
template <typename Bvh, typename Derived>
struct NodeIntersector {
    using Scalar = typename Bvh::ScalarType;

    std::array<int, 3> octant;

    NodeIntersector(const Ray<Scalar>& ray)
        : octant {
            std::signbit(ray.direction[0]),
            std::signbit(ray.direction[1]),
            std::signbit(ray.direction[2])
        }
    {}

    template <bool IsMin>
    bvh_always_inline
    Scalar intersect_axis(int axis, Scalar p, const Ray<Scalar>& ray, bool low_precision) const {
        return static_cast<const Derived*>(this)->template intersect_axis<IsMin>(axis, p, ray, low_precision);
    }

    bvh_always_inline
    std::pair<Scalar, Scalar> intersect(const typename Bvh::Node& node, const Ray<Scalar>& ray, bool low_precision) const {
        Vector3<Scalar> entry, exit;
        entry[0] = intersect_axis<true >(0, node.bounds[0 * 2 +     octant[0]], ray, low_precision);
        entry[1] = intersect_axis<true >(1, node.bounds[1 * 2 +     octant[1]], ray, low_precision);
        entry[2] = intersect_axis<true >(2, node.bounds[2 * 2 +     octant[2]], ray, low_precision);
        exit [0] = intersect_axis<false>(0, node.bounds[0 * 2 + 1 - octant[0]], ray, low_precision);
        exit [1] = intersect_axis<false>(1, node.bounds[1 * 2 + 1 - octant[1]], ray, low_precision);
        exit [2] = intersect_axis<false>(2, node.bounds[2 * 2 + 1 - octant[2]], ray, low_precision);
        // Note: This order for the min/max operations is guaranteed not to produce NaNs
        return std::make_pair(
            robust_max(entry[0], robust_max(entry[1], robust_max(entry[2], ray.tmin))),
            robust_min(exit [0], robust_min(exit [1], robust_min(exit [2], ray.tmax))));
    }

protected:
    ~NodeIntersector() {}
};

/// Fully robust ray-node intersection algorithm (see "Robust BVH Ray Traversal", by T. Ize).
template <typename Bvh>
struct RobustNodeIntersector : public NodeIntersector<Bvh, RobustNodeIntersector<Bvh>> {
    using Scalar = typename Bvh::ScalarType;

    // Padded inverse direction to avoid false-negatives in the ray-node test.
    Vector3<Scalar> padded_inverse_direction;
    Vector3<Scalar> inverse_direction;

    RobustNodeIntersector(const Ray<Scalar>& ray)
        : NodeIntersector<Bvh, RobustNodeIntersector<Bvh>>(ray)
    {
        inverse_direction = ray.direction.inverse();
        padded_inverse_direction = Vector3<Scalar>(
            add_ulp_magnitude(inverse_direction[0], 2),
            add_ulp_magnitude(inverse_direction[1], 2),
            add_ulp_magnitude(inverse_direction[2], 2));
    }

    template <bool IsMin>
    bvh_always_inline
    Scalar intersect_axis(int axis, Scalar p, const Ray<Scalar>& ray, bool) const {
        return (p - ray.origin[axis]) * (IsMin ? inverse_direction[axis] : padded_inverse_direction[axis]);
    }

    using NodeIntersector<Bvh, RobustNodeIntersector<Bvh>>::intersect;
};

/// Semi-robust, fast ray-node intersection algorithm.
template <typename Bvh>
struct FastNodeIntersector : public NodeIntersector<Bvh, FastNodeIntersector<Bvh>> {
    using Scalar = typename Bvh::ScalarType;

    Vector3<Scalar> scaled_origin;
    Vector3<Scalar> inverse_direction;

    FastNodeIntersector(const Ray<Scalar>& ray)
        : NodeIntersector<Bvh, FastNodeIntersector<Bvh>>(ray) 
    {
        inverse_direction = ray.direction.safe_inverse();
        scaled_origin     = -ray.origin * inverse_direction;
    }

    template <bool>
    bvh_always_inline
    Scalar intersect_axis(int axis, Scalar p, const Ray<Scalar>&, bool) const {
        return fast_multiply_add(p, inverse_direction[axis], scaled_origin[axis]);
    }

    using NodeIntersector<Bvh, FastNodeIntersector<Bvh>>::intersect;
};

/// Multiprecision intersection algorithm.
template <typename Bvh, mpfr_prec_t mantissa_width, mpfr_exp_t exponent_width>
struct MPNodeIntersector : public NodeIntersector<Bvh, MPNodeIntersector<Bvh, mantissa_width, exponent_width>> {
    using Scalar = typename Bvh::ScalarType;

    static constexpr mpfr_exp_t exponent_min = -(1 << (exponent_width - 1)) + 2;
    static constexpr mpfr_exp_t exponent_max = (1 << (exponent_width - 1));

    static bool initialized;
    static mpfr_t origin_d[3];
    static mpfr_t origin_u[3];
    static mpfr_t direction_d[3];
    static mpfr_t direction_u[3];
    static mpfr_t tmp;

    Vector3<Scalar> scaled_origin;
    Vector3<Scalar> inverse_direction;

    void check_exponent_and_set_inf(mpfr_t &num) const {
        if (mpfr_number_p(num)) {
            if (mpfr_get_exp(num) < exponent_min) mpfr_set_zero(num, mpfr_signbit(num) ? -1 : 1);
            else if (mpfr_get_exp(num) > exponent_max) mpfr_set_inf(num, mpfr_signbit(num) ? -1 : 1);
        }
    }

    MPNodeIntersector(const Ray<Scalar>& ray)
            : NodeIntersector<Bvh, MPNodeIntersector<Bvh, mantissa_width, exponent_width>>(ray)
    {
        if (!initialized) {
            for (int i = 0; i < 3; i++) {
                mpfr_init2(origin_d[i], mantissa_width + 1);
                mpfr_init2(origin_u[i], mantissa_width + 1);
                mpfr_init2(direction_d[i], mantissa_width + 1);
                mpfr_init2(direction_u[i], mantissa_width + 1);
            }

            mpfr_init2(tmp, mantissa_width + 1);

            initialized = true;
        }

        inverse_direction = ray.direction.safe_inverse();
        scaled_origin     = -ray.origin * inverse_direction;

        for (int i = 0; i < 3; i++) {
            mpfr_set_d(origin_d[i], ray.origin[i], MPFR_RNDD);
            mpfr_set_d(origin_u[i], ray.origin[i], MPFR_RNDU);
            mpfr_set_d(direction_d[i], ray.direction[i], MPFR_RNDD);
            mpfr_set_d(direction_u[i], ray.direction[i], MPFR_RNDU);
            check_exponent_and_set_inf(origin_d[i]);
            check_exponent_and_set_inf(origin_u[i]);
            check_exponent_and_set_inf(direction_d[i]);
            check_exponent_and_set_inf(direction_u[i]);
        }
    }

    template <bool IsMin>
    bvh_always_inline
    Scalar intersect_axis(int axis, Scalar p, const Ray<Scalar>& ray, bool low_precision) const {
        if (!low_precision)
            return fast_multiply_add(p, inverse_direction[axis], scaled_origin[axis]);
        if (!std::signbit(ray.direction[axis])) {
            if (IsMin) {
                mpfr_set_d(tmp, p, MPFR_RNDD);
                check_exponent_and_set_inf(tmp);
                mpfr_sub(tmp, tmp, origin_u[axis], MPFR_RNDD);
                check_exponent_and_set_inf(tmp);
                mpfr_div(tmp, tmp, direction_u[axis], MPFR_RNDD);
                check_exponent_and_set_inf(tmp);
            } else {
                mpfr_set_d(tmp, p, MPFR_RNDU);
                check_exponent_and_set_inf(tmp);
                mpfr_sub(tmp, tmp, origin_d[axis], MPFR_RNDU);
                check_exponent_and_set_inf(tmp);
                mpfr_div(tmp, tmp, direction_d[axis], MPFR_RNDU);
                check_exponent_and_set_inf(tmp);
            }
        } else {
            if (IsMin) {
                mpfr_set_d(tmp, p, MPFR_RNDU);
                check_exponent_and_set_inf(tmp);
                mpfr_sub(tmp, tmp, origin_d[axis], MPFR_RNDU);
                check_exponent_and_set_inf(tmp);
                mpfr_div(tmp, tmp, direction_d[axis], MPFR_RNDD);
                check_exponent_and_set_inf(tmp);
            } else {
                mpfr_set_d(tmp, p, MPFR_RNDD);
                check_exponent_and_set_inf(tmp);
                mpfr_sub(tmp, tmp, origin_u[axis], MPFR_RNDD);
                check_exponent_and_set_inf(tmp);
                mpfr_div(tmp, tmp, direction_u[axis], MPFR_RNDU);
                check_exponent_and_set_inf(tmp);
            }
        }
        return mpfr_get_d(tmp, MPFR_RNDN);
    }
};

template <typename Bvh, mpfr_prec_t mantissa_width, mpfr_exp_t exponent_width>
bool MPNodeIntersector<Bvh, mantissa_width, exponent_width>::initialized = false;

template <typename Bvh, mpfr_prec_t mantissa_width, mpfr_exp_t exponent_width>
mpfr_t MPNodeIntersector<Bvh, mantissa_width, exponent_width>::origin_d[3];

template <typename Bvh, mpfr_prec_t mantissa_width, mpfr_exp_t exponent_width>
mpfr_t MPNodeIntersector<Bvh, mantissa_width, exponent_width>::origin_u[3];

template <typename Bvh, mpfr_prec_t mantissa_width, mpfr_exp_t exponent_width>
mpfr_t MPNodeIntersector<Bvh, mantissa_width, exponent_width>::direction_d[3];

template <typename Bvh, mpfr_prec_t mantissa_width, mpfr_exp_t exponent_width>
mpfr_t MPNodeIntersector<Bvh, mantissa_width, exponent_width>::direction_u[3];

template <typename Bvh, mpfr_prec_t mantissa_width, mpfr_exp_t exponent_width>
mpfr_t MPNodeIntersector<Bvh, mantissa_width, exponent_width>::tmp;

} // namespace bvh

#endif
