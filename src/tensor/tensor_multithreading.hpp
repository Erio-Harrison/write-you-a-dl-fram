#include <thread>
#include <mutex>
#include <vector>
#include <functional>
#include "tensor_advanced.hpp"

template<typename T, size_t Dim>
class MultithreadedTensor : public AdvancedTensor<T, Dim>{
    
public:
    MultithreadedTensor() : AdvancedTensor<T, Dim>() {}

public:
    static const size_t MIN_ELEMENTS_PER_THREAD = 1000;

public:
    using AdvancedTensor<T,Dim>::AdvancedTensor;

    void parallel_add(const MultithreadedTensor<T, Dim>& other) {
        if (this->shape() != other.shape()) {
            throw std::invalid_argument("Tensors must have the same shape for parallel_add");
        }

        size_t total_size = this->data_ptr_->size();
        size_t num_threads = std::thread::hardware_concurrency();
        size_t elements_per_thread = std::max(MIN_ELEMENTS_PER_THREAD, total_size / num_threads);
        num_threads = std::min(num_threads, (total_size + elements_per_thread - 1) / elements_per_thread);

        std::vector<std::thread> threads;
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_size);
            threads.emplace_back([this, &other, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    (*this->data_ptr_)[i] += (*other.data_ptr_)[i];
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

    template<size_t OtherDim>
    MultithreadedTensor<T, std::max(Dim, OtherDim)> parallel_broadcast_add(const Tensor<T, OtherDim>& other) const {
        std::array<size_t, std::max(Dim, OtherDim)> new_shape;
        this->compute_broadcast_shape(this->shape(), other.shape(), new_shape);
        
        MultithreadedTensor<T, std::max(Dim, OtherDim)> result(new_shape);
        
        size_t total_size = result.data_ptr_->size();
        size_t num_threads = std::thread::hardware_concurrency();
        size_t elements_per_thread = std::max(MIN_ELEMENTS_PER_THREAD, total_size / num_threads);
        num_threads = std::min(num_threads, (total_size + elements_per_thread - 1) / elements_per_thread);

        std::vector<std::thread> threads;
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_size);
            threads.emplace_back([this, &other, &result, start, end]() {
                std::array<size_t, std::max(Dim, OtherDim)> index;
                for (size_t i = start; i < end; ++i) {
                    size_t temp = i;
                    for (int j = std::max(Dim, OtherDim) - 1; j >= 0; --j) {
                        index[j] = temp % result.shape()[j];
                        temp /= result.shape()[j];
                    }
                    result(index) = this->get_value(*this, index, Dim) + this->get_value(other, index, OtherDim);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        return result;
    }

    T parallel_sum() const {
        size_t total_size = this->data_ptr_->size();
        size_t num_threads = std::thread::hardware_concurrency();
        size_t elements_per_thread = std::max(MIN_ELEMENTS_PER_THREAD, total_size / num_threads);
        num_threads = std::min(num_threads, (total_size + elements_per_thread - 1) / elements_per_thread);

        std::vector<T> partial_sums(num_threads, 0);
        std::vector<std::thread> threads;

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * elements_per_thread;
            size_t end = std::min(start + elements_per_thread, total_size);
            threads.emplace_back([this, &partial_sums, t, start, end]() {
                T sum = 0;
                for (size_t i = start; i < end; ++i) {
                    sum += (*this->data_ptr_)[i];
                }
                partial_sums[t] = sum;

            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        T total_sum = std::accumulate(partial_sums.begin(), partial_sums.end(), T(0));

        return total_sum;
    }

};

template<typename T, size_t Dim>
const size_t MultithreadedTensor<T, Dim>::MIN_ELEMENTS_PER_THREAD;