#ifndef BVH_SINGLE_RAY_TRAVERSAL_QUANT_HPP
#define BVH_SINGLE_RAY_TRAVERSAL_QUANT_HPP

#include <cassert>

#include "bvh/bvh.hpp"
#include "bvh/ray.hpp"
#include "bvh/node_intersectors.hpp"
#include "bvh/utilities.hpp"

namespace bvh {

/// Single ray traversal algorithm, using the provided ray-node intersector.
template <typename Bvh, size_t StackSize = 64, typename NodeIntersector = FastNodeIntersector<Bvh>>
class SingleRayTraverserQuant {
public:
    static constexpr size_t stack_size = StackSize;

private:
    using Scalar = typename Bvh::ScalarType;

    struct Stack {
        using Element = std::tuple<typename Bvh::IndexType, std::array<float, 3>, std::array<float, 3>>;

        Element elements[stack_size];
        size_t size = 0;

        void push(const Element& t) {
            assert(size < stack_size);
            elements[size++] = t;
        }

        Element pop() {
            assert(!empty());
            return elements[--size];
        }

        bool empty() const { return size == 0; }
    };

    template <typename PrimitiveIntersector, typename Statistics>
    bvh_always_inline
    std::optional<typename PrimitiveIntersector::Result>& intersect_leaf(
        const typename Bvh::Node& node,
        Ray<Scalar>& ray,
        std::optional<typename PrimitiveIntersector::Result>& best_hit,
        PrimitiveIntersector& primitive_intersector,
        Statistics& statistics) const
    {
        assert(node.is_leaf());
        size_t begin = node.first_child_or_primitive;
        size_t end   = begin + node.primitive_count;
        for (size_t i = begin; i < end; ++i) {
            if (auto hit = primitive_intersector.intersect(i, ray, statistics)) {
                best_hit = hit;
                if (primitive_intersector.any_hit)
                    return best_hit;
                ray.tmax = hit->distance();
            }
        }
        return best_hit;
    }

    template <typename PrimitiveIntersector, typename Statistics>
    bvh_always_inline
    std::optional<typename PrimitiveIntersector::Result>
    intersect(Ray<Scalar> ray, PrimitiveIntersector& primitive_intersector, Statistics& statistics) const {
        auto best_hit = std::optional<typename PrimitiveIntersector::Result>(std::nullopt);

        // If the root is a leaf, intersect it and return
        if (bvh_unlikely(bvh.nodes[0].is_leaf()))
            return intersect_leaf(bvh.nodes[0], ray, best_hit, primitive_intersector, statistics);

        NodeIntersector node_intersector(ray);

        // This traversal loop is eager, because it immediately processes leaves instead of pushing them on the stack.
        // This is generally beneficial for performance because intersections will likely be found which will
        // allow to cull more subtrees with the ray-box test of the traversal loop.
        Stack stack;
        std::array<float, 3> curr_min_bounds = {
            bvh.nodes[0].bounds[0],
            bvh.nodes[0].bounds[2],
            bvh.nodes[0].bounds[4],
        };
        std::array<float, 3> curr_exp = {
            bvh.nodes[0].exp[0],
            bvh.nodes[0].exp[1],
            bvh.nodes[0].exp[2]
        };
        auto* left_child = &bvh.nodes[bvh.nodes[0].first_child_or_primitive];
        while (true) {
            statistics.traversal_steps++;

            auto* right_child = left_child + 1;

            std::array<float, 6> left_bounds = {};
            std::array<float, 6> right_bounds = {};
            std::vector<std::pair<typename Bvh::Node*, float*>> child_bounds_pair = {
                { left_child, left_bounds.data() },
                { right_child, right_bounds.data() }
            };

            for (auto &[child, bounds] : child_bounds_pair) {
                for (int i = 0; i < 3; i++) {
                    bounds[i * 2] = curr_min_bounds[i] + child->bounds_quant[i * 2] * curr_exp[i];
                    bounds[i * 2 + 1] = curr_min_bounds[i] + child->bounds_quant[i * 2 + 1] * curr_exp[i];
                }
            }

            std::array<float, 3> left_min_bounds = {
                left_bounds[0],
                left_bounds[2],
                left_bounds[4]
            };
            std::array<float, 3> right_min_bounds = {
                right_bounds[0],
                right_bounds[2],
                right_bounds[4]
            };

            std::array<float, 3> left_exp = {
                left_child->exp[0],
                left_child->exp[1],
                left_child->exp[2]
            };
            std::array<float, 3> right_exp = {
                right_child->exp[0],
                right_child->exp[1],
                right_child->exp[2]
            };

            auto distance_left  = node_intersector.intersect(left_bounds.data(),  ray);
            auto distance_right = node_intersector.intersect(right_bounds.data(), ray);

            if (distance_left.first <= distance_left.second) {
                if (bvh_unlikely(left_child->is_leaf())) {
                    if (intersect_leaf(*left_child, ray, best_hit, primitive_intersector, statistics) &&
                        primitive_intersector.any_hit)
                        break;
                    left_child = nullptr;
                }
            } else
                left_child = nullptr;

            if (distance_right.first <= distance_right.second) {
                if (bvh_unlikely(right_child->is_leaf())) {
                    if (intersect_leaf(*right_child, ray, best_hit, primitive_intersector, statistics) &&
                        primitive_intersector.any_hit)
                        break;
                    right_child = nullptr;
                }
            } else
                right_child = nullptr;

            if (left_child) {
                if (right_child) {
                    statistics.both_intersected++;
                    if (distance_left.first > distance_right.first) {
                        std::swap(left_child, right_child);
                        std::swap(left_min_bounds, right_min_bounds);
                        std::swap(left_exp, right_exp);
                    }
                    stack.push({right_child->first_child_or_primitive, right_min_bounds, right_exp});
                }
                left_child = &bvh.nodes[left_child->first_child_or_primitive];
                curr_min_bounds = left_min_bounds;
                curr_exp = left_exp;
            } else if (right_child) {
                left_child = &bvh.nodes[right_child->first_child_or_primitive];
                curr_min_bounds = right_min_bounds;
                curr_exp = right_exp;
            } else {
                if (stack.empty())
                    break;
                auto [t1, t2, t3] = stack.pop();
                left_child = &bvh.nodes[t1];
                curr_min_bounds = t2;
                curr_exp = t3;
            }
        }

        if (best_hit.has_value())
            statistics.finalize++;
        return best_hit;
    }

    const Bvh& bvh;

public:
    /// Statistics collected during traversal.
    struct Statistics {
        size_t traversal_steps = 0;
        size_t both_intersected = 0;
        size_t intersections_a = 0;
        size_t intersections_b = 0;
        size_t finalize = 0;
    };

    SingleRayTraverserQuant(const Bvh& bvh)
        : bvh(bvh)
    {}

    /// Intersects the BVH with the given ray and intersector.
    template <typename PrimitiveIntersector>
    bvh_always_inline
    std::optional<typename PrimitiveIntersector::Result>
    traverse(const Ray<Scalar>& ray, PrimitiveIntersector& intersector) const {
        struct {
            struct Empty {
                Empty& operator ++ (int)    { return *this; }
                Empty& operator ++ ()       { return *this; }
                Empty& operator += (size_t) { return *this; }
            } traversal_steps, both_intersected, intersections_a, intersections_b, finalize;
        } statistics;
        return intersect(ray, intersector, statistics);
    }

    /// Intersects the BVH with the given ray and intersector.
    /// Record statistics on the number of traversal and intersection steps.
    template <typename PrimitiveIntersector>
    bvh_always_inline
    std::optional<typename PrimitiveIntersector::Result>
    traverse(const Ray<Scalar>& ray, PrimitiveIntersector& primitive_intersector, Statistics& statistics) const {
        return intersect(ray, primitive_intersector, statistics);
    }
};

} // namespace bvh

#endif
