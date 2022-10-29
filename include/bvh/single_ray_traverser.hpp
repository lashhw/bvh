#ifndef BVH_SINGLE_RAY_TRAVERSAL_HPP
#define BVH_SINGLE_RAY_TRAVERSAL_HPP

#include <bitset>
#include <cassert>
#include <stack>

#include "bvh/bvh.hpp"
#include "bvh/ray.hpp"
#include "bvh/node_intersectors.hpp"
#include "bvh/utilities.hpp"

namespace bvh {

/// Single ray traversal algorithm, using the provided ray-node intersector.
template <typename Bvh, size_t StackSize = 64, typename NodeIntersector = FastNodeIntersector<Bvh>>
class SingleRayTraverser {
public:
    static constexpr size_t stack_size = StackSize;

private:
    using Scalar = typename Bvh::ScalarType;

    struct Stack {
        using Element = typename Bvh::IndexType;

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

    struct StackElement {
        size_t node_index;
        size_t depth;
        bool single;
    };

    template <typename PrimitiveIntersector, typename Statistics>
    bvh_always_inline
    std::optional<typename PrimitiveIntersector::Result> intersect_leaf(
        const typename Bvh::Node& node,
        Ray<Scalar>& ray,
        std::optional<typename PrimitiveIntersector::Result>& best_hit,
        PrimitiveIntersector& primitive_intersector,
        Statistics& statistics) const
    {
        assert(node.is_leaf());
        size_t begin = node.first_child_or_primitive;
        size_t end   = begin + node.primitive_count;
        statistics.trig_intersections += end - begin;
        bool closer = false;
        for (size_t i = begin; i < end; ++i) {
            if (auto hit = primitive_intersector.intersect(i, ray)) {
                closer = true;
                best_hit = hit;
                if (primitive_intersector.any_hit)
                    return best_hit;
                ray.tmax = hit->distance();
            }
        }
        return closer ? best_hit : std::nullopt;
    }

    template <typename PrimitiveIntersector, typename Statistics>
    bvh_always_inline
    std::optional<typename PrimitiveIntersector::Result>
    intersect(Ray<Scalar> ray, PrimitiveIntersector& primitive_intersector, Statistics& statistics,
              bool follow_first_path, std::bitset<64> first_path,
              std::bitset<64> &closest_hit_path, size_t &closest_hit_depth) const {
        auto best_hit = std::optional<typename PrimitiveIntersector::Result>(std::nullopt);

        // If the root is a leaf, intersect it and return
        if (bvh_unlikely(bvh.nodes[0].is_leaf()))
            return intersect_leaf(bvh.nodes[0], ray, best_hit, primitive_intersector, statistics);

        statistics.node_traversed++;

        NodeIntersector node_intersector(ray);

        // This traversal loop is eager, because it immediately processes leaves instead of pushing them on the stack.
        // This is generally beneficial for performance because intersections will likely be found which will
        // allow to cull more subtrees with the ray-box test of the traversal loop.
        std::stack<StackElement> stack;
        auto* curr_node = &bvh.nodes[0];
        auto* left_child = &bvh.nodes[bvh.nodes[0].first_child_or_primitive];
        bool curr_single = false;

        while (follow_first_path) {
            typename Bvh::Node* other_node;
            if (first_path.test(0)) {
                curr_node = &bvh.nodes[curr_node->first_child_or_primitive + 1];
                other_node = curr_node - 1;
            } else {
                curr_node = &bvh.nodes[curr_node->first_child_or_primitive];
                other_node = curr_node + 1;
            }
            first_path >>= 1;

            statistics.node_traversed++;
            stack.push({size_t(other_node - &bvh.nodes[0]), 0, true});

            if (curr_node->is_leaf()) {
                intersect_leaf(*curr_node, ray, best_hit, primitive_intersector, statistics);

                auto stack_top = stack.top();
                stack.pop();
                left_child = &bvh.nodes[stack_top.node_index];
                curr_single = true;

                follow_first_path = false;
            }
        }

        std::bitset<64> curr_path;
        size_t curr_depth = 0;

        while (true) {
            auto* right_child = left_child + 1;
            auto distance_left  = node_intersector.intersect(*left_child,  ray);
            statistics.node_traversed++;
            statistics.node_intersections++;
            std::pair<typename NodeIntersector::Scalar, typename NodeIntersector::Scalar> distance_right;
            if (!curr_single) {
                distance_right = node_intersector.intersect(*right_child, ray);
                statistics.node_traversed++;
                statistics.node_intersections++;
            }

            if (distance_left.first <= distance_left.second) {
                if (bvh_unlikely(left_child->is_leaf())) {
                    if (intersect_leaf(*left_child, ray, best_hit, primitive_intersector, statistics)) {
                        if (primitive_intersector.any_hit) break;
                        closest_hit_path = curr_path;
                        closest_hit_path.reset(curr_depth);
                        closest_hit_depth = curr_depth + 1;
                    }
                    left_child = nullptr;
                }
            } else
                left_child = nullptr;

            if (!curr_single && distance_right.first <= distance_right.second) {
                if (bvh_unlikely(right_child->is_leaf())) {
                    if (intersect_leaf(*right_child, ray, best_hit, primitive_intersector, statistics)) {
                        if (primitive_intersector.any_hit) break;
                        closest_hit_path = curr_path;
                        closest_hit_path.set(curr_depth);
                        closest_hit_depth = curr_depth + 1;
                    }
                    right_child = nullptr;
                }
            } else
                right_child = nullptr;

            if (left_child) {
                curr_path.reset(curr_depth);
                if (right_child) {
                    if (distance_left.first > distance_right.first) {
                        std::swap(left_child, right_child);
                        curr_path.set(curr_depth);
                    }
                    stack.push({right_child->first_child_or_primitive, curr_depth, false});
                }
                left_child = &bvh.nodes[left_child->first_child_or_primitive];
                curr_single = false;
            } else if (right_child) {
                left_child = &bvh.nodes[right_child->first_child_or_primitive];
                curr_path.set(curr_depth);
                curr_single = false;
            } else {
                if (stack.empty())
                    break;
                auto stack_top = stack.top();
                stack.pop();
                left_child = &bvh.nodes[stack_top.node_index];
                curr_depth = stack_top.depth;
                curr_single = stack_top.single;
                curr_path.flip(curr_depth);
            }
            curr_depth++;
        }

        return best_hit;
    }

    const Bvh& bvh;

public:
    /// Statistics collected during traversal.
    struct Statistics {
        size_t node_traversed     = 0;
        size_t node_intersections = 0;
        size_t trig_intersections = 0;
    };

    SingleRayTraverser(const Bvh& bvh)
        : bvh(bvh)
    {}

    /// Intersects the BVH with the given ray and intersector.
    template <typename PrimitiveIntersector>
    bvh_always_inline
    std::optional<typename PrimitiveIntersector::Result>
    traverse(const Ray<Scalar>& ray, PrimitiveIntersector& intersector,
             bool follow_first_path, std::bitset<64> first_path,
             std::bitset<64> &closest_hit_path, size_t &closest_hit_depth) const {
        struct {
            struct Empty {
                Empty& operator ++ (int)    { return *this; }
                Empty& operator ++ ()       { return *this; }
                Empty& operator += (size_t) { return *this; }
            } node_traversed, node_intersections, trig_intersections;
        } statistics;
        return intersect(ray, intersector, statistics,
                         follow_first_path, first_path,
                         closest_hit_path, closest_hit_depth);
    }

    /// Intersects the BVH with the given ray and intersector.
    /// Record statistics on the number of traversal and intersection steps.
    template <typename PrimitiveIntersector>
    bvh_always_inline
    std::optional<typename PrimitiveIntersector::Result>
    traverse(const Ray<Scalar>& ray, PrimitiveIntersector& primitive_intersector, Statistics& statistics,
             bool follow_first_path, std::bitset<64> first_path,
             std::bitset<64> &closest_hit_path, size_t &closest_hit_depth) const {
        return intersect(ray, primitive_intersector, statistics,
                         follow_first_path, first_path,
                         closest_hit_path, closest_hit_depth);
    }
};

} // namespace bvh

#endif
