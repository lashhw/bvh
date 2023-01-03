#ifndef BVH_MIXED_SWEEP_SAH_BUILDER_HPP
#define BVH_MIXED_SWEEP_SAH_BUILDER_HPP

#include <array>
#include <optional>
#include <mpfr.h>

#include "bvh/bvh.hpp"
#include "bvh/bounding_box.hpp"
#include "bvh/top_down_builder.hpp"
#include "bvh/sah_based_algorithm.hpp"
#include "bvh/radix_sort.hpp"

namespace bvh {

template <typename, size_t, size_t> class MixedSweepSahBuildTask;

/// This is a top-down, full-sweep SAH-based BVH builder. Primitives are only
/// sorted once, and a stable partitioning algorithm is used when splitting,
/// so as to keep the relative order of primitives within each partition intact.
template <typename Bvh, size_t mantissa_width, size_t exponent_width>
class MixedSweepSahBuilder : public TopDownBuilder, public SahBasedAlgorithm<Bvh> {
    using Scalar    = typename Bvh::ScalarType;
    using BuildTask = MixedSweepSahBuildTask<Bvh, mantissa_width, exponent_width>;
    using Key       = typename SizedIntegerType<sizeof(Scalar) * CHAR_BIT>::Unsigned;
    using Mark      = typename BuildTask::MarkType;

    using TopDownBuilder::run_task;

    friend BuildTask;

    RadixSort<10> radix_sort;
    Bvh& bvh;

public:
    using TopDownBuilder::max_depth;
    using TopDownBuilder::max_leaf_size;

    Scalar high_traversal_cost;
    Scalar low_traversal_cost;
    Scalar k;

    MixedSweepSahBuilder(Bvh& bvh, Scalar high_traversal_cost, Scalar low_traversal_cost, Scalar k)
        : bvh(bvh), high_traversal_cost(high_traversal_cost), low_traversal_cost(low_traversal_cost), k(k)
    {}

    void build(
        const BoundingBox<Scalar>& global_bbox,
        const BoundingBox<Scalar>* bboxes,
        const Vector3<Scalar>* centers,
        size_t primitive_count)
    {
        assert(primitive_count > 0);

        // Allocate buffers
        bvh.nodes = std::make_unique<typename Bvh::Node[]>(2 * primitive_count + 1);
        bvh.primitive_indices = std::make_unique<size_t[]>(primitive_count);

        auto reference_data = std::make_unique<size_t[]>(primitive_count * 3);
        auto high_cost_data = std::make_unique<Scalar[]>(primitive_count * 3);
        auto low_cost_data  = std::make_unique<Scalar[]>(primitive_count * 3);
        auto key_data       = std::make_unique<Key[]>(primitive_count * 2);
        auto mark_data      = std::make_unique<Mark[]>(primitive_count);

        std::array<Scalar*, 3> high_costs = {
            high_cost_data.get(),
            high_cost_data.get() + primitive_count,
            high_cost_data.get() + 2 * primitive_count
        };

        std::array<Scalar*, 3> low_costs = {
            low_cost_data.get(),
            low_cost_data.get() + primitive_count,
            low_cost_data.get() + 2 * primitive_count
        };

        std::array<size_t*, 3> sorted_references;
        size_t* unsorted_references = bvh.primitive_indices.get();
        Key* sorted_keys = key_data.get();
        Key* unsorted_keys = key_data.get() + primitive_count;

        bvh.node_count = 1;
        bvh.nodes[0].bounding_box_proxy() = global_bbox;
        bvh.nodes[0].high_precision = true;

        #pragma omp parallel
        {
            // Sort the primitives on each axis once
            for (int axis = 0; axis < 3; ++axis) {
                #pragma omp single
                {
                    sorted_references[axis] = unsorted_references;
                    unsorted_references = reference_data.get() + axis * primitive_count;
                    // Make sure that one array is the final array of references used by the BVH
                    if (axis != 0 && sorted_references[axis] == bvh.primitive_indices.get())
                        std::swap(sorted_references[axis], unsorted_references);
                    assert(axis < 2 ||
                           sorted_references[0] == bvh.primitive_indices.get() ||
                           sorted_references[1] == bvh.primitive_indices.get());
                }

                #pragma omp for
                for (size_t i = 0; i < primitive_count; ++i) {
                    sorted_keys[i] = radix_sort.make_key(centers[i][axis]);
                    sorted_references[axis][i] = i;
                }

                radix_sort.sort_in_parallel(
                    sorted_keys,
                    unsorted_keys,
                    sorted_references[axis],
                    unsorted_references,
                    primitive_count,
                    sizeof(Scalar) * CHAR_BIT);
            }

            #pragma omp single
            {
                BuildTask first_task(*this, bboxes, centers, sorted_references, high_costs, low_costs, k, mark_data.get());
                run_task(first_task, 0, 0, primitive_count, 0);
            }
        }
    }
};

template <typename Bvh, size_t mantissa_width, size_t exponent_width>
class MixedSweepSahBuildTask : public TopDownBuildTask {
    using Scalar  = typename Bvh::ScalarType;
    using IndexType = typename Bvh::IndexType;
    using Builder = MixedSweepSahBuilder<Bvh, mantissa_width, exponent_width>;
    using Mark    = uint_fast8_t;

    using TopDownBuildTask::WorkItem;

    static constexpr mp_exp_t exponent_min = -(1 << (exponent_width - 1)) + 2;
    static constexpr mp_exp_t exponent_max = (1 << (exponent_width - 1));

    Builder& builder;
    const BoundingBox<Scalar>* bboxes;
    const Vector3<Scalar>* centers;

    std::array<size_t* bvh_restrict, 3> references;
    std::array<Scalar* bvh_restrict, 3> high_costs;
    std::array<Scalar* bvh_restrict, 3> low_costs;
    Scalar k;
    Mark* marks;

    void check_exponent_and_set_inf(mpfr_t &num) const {
        if (mpfr_number_p(num)) {
            if (mpfr_get_exp(num) < exponent_min) mpfr_set_zero(num, mpfr_signbit(num) ? -1 : 1);
            else if (mpfr_get_exp(num) > exponent_max) mpfr_set_inf(num, mpfr_signbit(num) ? -1 : 1);
        }
    }

    void set_low_bbox(BoundingBox<Scalar> &low_bbox, mpfr_t &tmp, const BoundingBox<Scalar> &high_bbox) {
        for (int i = 0; i < 3; i++) {
            mpfr_set_flt(tmp, high_bbox.min[i], MPFR_RNDD);
            check_exponent_and_set_inf(tmp);
            low_bbox.min[i] = mpfr_get_flt(tmp, MPFR_RNDD);
            mpfr_set_flt(tmp, high_bbox.max[i], MPFR_RNDU);
            check_exponent_and_set_inf(tmp);
            low_bbox.max[i] = mpfr_get_flt(tmp, MPFR_RNDU);
        }
    }

    std::tuple<Scalar, size_t, bool, bool> find_split(int axis, size_t begin, size_t end,
                                                      const BoundingBox<Scalar> &parent_bbox) {
        MPFR_DECL_INIT(tmp, mantissa_width + 1);
        auto parent_bbox_half_area = parent_bbox.half_area();

        auto bbox = BoundingBox<Scalar>::empty();
        for (size_t i = end - 1; i > begin; --i) {
            bbox.extend(bboxes[references[axis][i]]);
            high_costs[axis][i] = builder.high_traversal_cost +
                                  bbox.half_area() / parent_bbox_half_area * static_cast<Scalar>(end - i);

            BoundingBox<Scalar> low_bbox;
            set_low_bbox(low_bbox, tmp, bbox);
            low_bbox.shrink(parent_bbox);
            low_costs[axis][i] = builder.low_traversal_cost +
                                 k * low_bbox.half_area() / parent_bbox_half_area * static_cast<Scalar>(end - i);
        }

        bbox = BoundingBox<Scalar>::empty();
        auto best_split = std::tuple<Scalar, size_t, bool, bool>(std::numeric_limits<Scalar>::max(), end, false, false);
        for (size_t i = begin; i < end - 1; ++i) {
            bbox.extend(bboxes[references[axis][i]]);
            auto high_cost = builder.high_traversal_cost +
                             bbox.half_area() / parent_bbox_half_area * static_cast<Scalar>(i + 1 - begin);

            BoundingBox<Scalar> low_bbox;
            set_low_bbox(low_bbox, tmp, bbox);
            low_bbox.shrink(parent_bbox);
            auto low_cost = builder.low_traversal_cost +
                            k * low_bbox.half_area() / parent_bbox_half_area * static_cast<Scalar>(i + 1 - begin);

            bool left_high;
            Scalar left_cost;
            if (high_cost < low_cost) {
                left_high = true;
                left_cost = high_cost;
            } else {
                left_high = false;
                left_cost = low_cost;
            }

            bool right_high;
            Scalar right_cost;
            if (high_costs[axis][i + 1] < low_costs[axis][i + 1]) {
                right_high = true;
                right_cost = high_costs[axis][i + 1];
            } else {
                right_high = false;
                right_cost = low_costs[axis][i + 1];
            }

            if (left_cost + right_cost < std::get<0>(best_split))
                best_split = std::make_tuple(left_cost + right_cost, i + 1, left_high, right_high);
        }
        return best_split;
    }

public:
    using MarkType     = Mark;
    using WorkItemType = WorkItem;

    MixedSweepSahBuildTask(
        Builder& builder,
        const BoundingBox<Scalar>* bboxes,
        const Vector3<Scalar>* centers,
        const std::array<size_t*, 3>& references,
        const std::array<Scalar*, 3>& high_costs,
        const std::array<Scalar*, 3>& low_costs,
        Scalar k,
        Mark* marks)
        : builder(builder)
        , bboxes(bboxes)
        , centers(centers)
        , references { references[0], references[1], references[2] }
        , high_costs { high_costs[0], high_costs[1], high_costs[2] }
        , low_costs { low_costs[0], low_costs[1], low_costs[2] }
        , k(k)
        , marks(marks)
    {}

    std::optional<std::pair<WorkItem, WorkItem>> build(const WorkItem& item) {
        auto& bvh  = builder.bvh;
        auto& node = bvh.nodes[item.node_index];

        auto make_leaf = [] (typename Bvh::Node& node, size_t begin, size_t end) {
            node.first_child_or_primitive = static_cast<IndexType>(begin);
            node.primitive_count          = static_cast<IndexType>(end - begin);
        };

        if (item.work_size() <= 1 || item.depth >= builder.max_depth) {
            make_leaf(node, item.begin, item.end);
            return std::nullopt;
        }

        std::tuple<Scalar, size_t, bool, bool> best_splits[3];
        [[maybe_unused]] bool should_spawn_tasks = item.work_size() > builder.task_spawn_threshold;

        // Sweep primitives to find the best cost
        MPFR_DECL_INIT(tmp, mantissa_width + 1);
        BoundingBox<Scalar> parent_bbox;
        parent_bbox.min = { node.bounds[0], node.bounds[2], node.bounds[4] };
        parent_bbox.max = { node.bounds[1], node.bounds[3], node.bounds[5] };
        if (!node.high_precision)
            set_low_bbox(parent_bbox, tmp, parent_bbox);
        #pragma omp taskloop if (should_spawn_tasks) grainsize(1) default(shared)
        for (int axis = 0; axis < 3; ++axis)
            best_splits[axis] = find_split(axis, item.begin, item.end, parent_bbox);

        unsigned best_axis = 0;
        if (std::get<0>(best_splits[0]) > std::get<0>(best_splits[1]))
            best_axis = 1;
        if (std::get<0>(best_splits[best_axis]) > std::get<0>(best_splits[2]))
            best_axis = 2;

        auto split_index = std::get<1>(best_splits[best_axis]);

        // Make sure the cost of splitting does not exceed the cost of not splitting
        auto max_split_cost = static_cast<Scalar>(item.work_size());
        if (std::get<0>(best_splits[best_axis]) >= max_split_cost) {
            if (item.work_size() > builder.max_leaf_size) {
                // Fallback strategy: median split on largest axis
                best_axis = node.bounding_box_proxy().to_bounding_box().largest_axis();
                split_index = (item.begin + item.end) / 2;
            } else {
                make_leaf(node, item.begin, item.end);
                return std::nullopt;
            }
        }

        unsigned other_axis[2] = { (best_axis + 1) % 3, (best_axis + 2) % 3 };

        for (size_t i = item.begin;  i < split_index; ++i) marks[references[best_axis][i]] = 1;
        for (size_t i = split_index; i < item.end;    ++i) marks[references[best_axis][i]] = 0;
        auto partition_predicate = [&] (size_t i) { return marks[i] != 0; };

        auto left_bbox  = BoundingBox<Scalar>::empty();
        auto right_bbox = BoundingBox<Scalar>::empty();

        // Partition reference arrays and compute bounding boxes
        #pragma omp taskgroup
        {
            #pragma omp task if (should_spawn_tasks) default(shared)
            { std::stable_partition(references[other_axis[0]] + item.begin, references[other_axis[0]] + item.end, partition_predicate); }
            #pragma omp task if (should_spawn_tasks) default(shared)
            { std::stable_partition(references[other_axis[1]] + item.begin, references[other_axis[1]] + item.end, partition_predicate); }
            #pragma omp task if (should_spawn_tasks) default(shared)
            {
                for (size_t i = item.begin; i < split_index; ++i)
                    left_bbox.extend(bboxes[references[best_axis][i]]);
            }
            #pragma omp task if (should_spawn_tasks) default(shared)
            {
                for (size_t i = split_index; i < item.end; ++i)
                    right_bbox.extend(bboxes[references[best_axis][i]]);
            }
        }

        // Allocate space for children
        size_t first_child;
        #pragma omp atomic capture
        { first_child = bvh.node_count; bvh.node_count += 2; }

        auto& left  = bvh.nodes[first_child + 0];
        auto& right = bvh.nodes[first_child + 1];
        node.first_child_or_primitive = static_cast<IndexType>(first_child);
        node.primitive_count          = 0;

        left.bounding_box_proxy()  = left_bbox;
        left.high_precision = std::get<2>(best_splits[best_axis]);
        right.bounding_box_proxy() = right_bbox;
        right.high_precision = std::get<3>(best_splits[best_axis]);
        WorkItem first_item (first_child + 0, item.begin, split_index, item.depth + 1);
        WorkItem second_item(first_child + 1, split_index, item.end,   item.depth + 1);
        return std::make_optional(std::make_pair(first_item, second_item));
    }
};

} // namespace bvh

#endif
