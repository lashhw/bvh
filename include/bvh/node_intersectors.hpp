#ifndef BVH_NODE_INTERSECTORS_HPP
#define BVH_NODE_INTERSECTORS_HPP

#include <cmath>

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
    Scalar intersect_axis(int axis, Scalar p, const Ray<Scalar>& ray) const {
        return static_cast<const Derived*>(this)->template intersect_axis<IsMin>(axis, p, ray);
    }

    bvh_always_inline
    std::pair<Scalar, Scalar> intersect(const typename Bvh::Node& node, const Ray<Scalar>& ray) const {
        Vector3<Scalar> entry, exit;
        entry[0] = intersect_axis<true >(0, node.bounds[0 * 2 +     octant[0]], ray);
        entry[1] = intersect_axis<true >(1, node.bounds[1 * 2 +     octant[1]], ray);
        entry[2] = intersect_axis<true >(2, node.bounds[2 * 2 +     octant[2]], ray);
        exit [0] = intersect_axis<false>(0, node.bounds[0 * 2 + 1 - octant[0]], ray);
        exit [1] = intersect_axis<false>(1, node.bounds[1 * 2 + 1 - octant[1]], ray);
        exit [2] = intersect_axis<false>(2, node.bounds[2 * 2 + 1 - octant[2]], ray);
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
    Scalar intersect_axis(int axis, Scalar p, const Ray<Scalar>& ray) const {
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
    Scalar intersect_axis(int axis, Scalar p, const Ray<Scalar>&) const {
        return fast_multiply_add(p, inverse_direction[axis], scaled_origin[axis]);
    }

    using NodeIntersector<Bvh, FastNodeIntersector<Bvh>>::intersect;
};

/// Multiprecision intersection algorithm.
template <typename Bvh, size_t mantissa_width, size_t exponent_width>
struct MPNodeIntersector : public NodeIntersector<Bvh, MPNodeIntersector<Bvh, mantissa_width, exponent_width>> {
    using Scalar = typename Bvh::ScalarType;

    mp_exp_t exponent_min;
    mp_exp_t exponent_max;

    mutable mpfr_t tmp;

    mpfr_t origin_u[3];
    mpfr_t origin_d[3];
    mpfr_t direction_u[3];
    mpfr_t direction_d[3];

    MPNodeIntersector(const Ray<Scalar>& ray)
            : NodeIntersector<Bvh, MPNodeIntersector<Bvh, mantissa_width, exponent_width>>(ray)
    {
        exponent_max = (1 << (exponent_width - 1));
        exponent_min = -exponent_max + 2;

        mpfr_init2(tmp, mantissa_width);

        mpfr_t one;
        mpfr_init2(one, mantissa_width);
        mpfr_set_str(one, "1", 2, MPFR_RNDN);

        mpfr_t next;
        mpfr_init2(next, mantissa_width);
        mpfr_set(next, one, MPFR_RNDN);
        mpfr_nextabove(next);

        mpfr_t epsilon;
        mpfr_init2(epsilon, mantissa_width);
        mpfr_sub(epsilon, next, one, MPFR_RNDN);

        for (int i = 0; i < 3; i++) {
            mpfr_init2(origin_u[i], mantissa_width);
            mpfr_set_d(origin_u[i], ray.origin[i], MPFR_RNDU);

            mpfr_init2(origin_d[i], mantissa_width);
            mpfr_set_d(origin_d[i], ray.origin[i], MPFR_RNDD);

            mpfr_init2(direction_u[i], mantissa_width);
            mpfr_set_d(direction_u[i], ray.direction[i], MPFR_RNDU);

            mpfr_init2(direction_d[i], mantissa_width);
            mpfr_set_d(direction_d[i], ray.direction[i], MPFR_RNDD);

            mpfr_t abs_ray_direction;
            mpfr_init2(abs_ray_direction, mantissa_width);
            if (!std::signbit(ray.direction[i]))
                mpfr_abs(abs_ray_direction, direction_d[i], MPFR_RNDD);
            else
                mpfr_abs(abs_ray_direction, direction_u[i], MPFR_RNDD);

            if (mpfr_cmp(abs_ray_direction, epsilon) < 0) {
                mpfr_set(direction_u[i], epsilon, MPFR_RNDN);
                mpfr_set(direction_d[i], epsilon, MPFR_RNDN);
            }
        }
    }

    template <bool IsMin>
    bvh_always_inline
    Scalar intersect_axis(int axis, Scalar p, const Ray<Scalar>& ray) const {
        if (!std::signbit(ray.direction[axis])) {
            if (IsMin) {
                mpfr_set_d(tmp, p, MPFR_RNDD);
                mpfr_sub(tmp, tmp, origin_u[axis], MPFR_RNDD);
                mpfr_div(tmp, tmp, direction_u[axis], MPFR_RNDD);
            } else {
                mpfr_set_d(tmp, p, MPFR_RNDU);
                mpfr_sub(tmp, tmp, origin_d[axis], MPFR_RNDU);
                mpfr_div(tmp, tmp, direction_d[axis], MPFR_RNDU);
            }
        } else {
            if (IsMin) {
                mpfr_set_d(tmp, p, MPFR_RNDU);
                mpfr_sub(tmp, tmp, origin_d[axis], MPFR_RNDU);
                mpfr_div(tmp, tmp, direction_d[axis], MPFR_RNDD);
            } else {
                mpfr_set_d(tmp, p, MPFR_RNDD);
                mpfr_sub(tmp, tmp, origin_u[axis], MPFR_RNDD);
                mpfr_div(tmp, tmp, direction_u[axis], MPFR_RNDU);
            }
        }
        return mpfr_get_d(tmp, MPFR_RNDN);
    }
};

} // namespace bvh

#endif
