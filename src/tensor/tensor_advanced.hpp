#include <algorithm>
#include <numeric>
#include <immintrin.h>
#include "tensor.hpp"

template <typename T, size_t Dim>
class AdvancedTensor : public Tensor<T,Dim>{
public:
    AdvancedTensor() : Tensor<T, Dim>() {}

public:
    using Tensor<T, Dim>::Tensor;
    using Tensor<T, Dim>::operator+;
    using Tensor<T, Dim>::operator-;
    using Tensor<T, Dim>::operator*;
    using Tensor<T, Dim>::operator/;

    AdvancedTensor(const std::array<size_t, Dim>& shape) : Tensor<T, Dim>(shape) {}
    AdvancedTensor(const std::array<size_t, Dim>& shape, const std::vector<T>& data) : Tensor<T, Dim>(shape, data) {}

    template<size_t OtherDim>
    AdvancedTensor<T, std::max(Dim, OtherDim)> broadcast_add(const Tensor<T, OtherDim>& other) const {
        std::array<size_t, std::max(Dim, OtherDim)> new_shape;
        compute_broadcast_shape(this->shape(), other.shape(), new_shape);
        
        AdvancedTensor<T, std::max(Dim, OtherDim)> result(new_shape);
        
        broadcast_add_impl(*this, other, result, std::array<size_t, std::max(Dim, OtherDim)>(), 0);
        
        return result;
    }

    AdvancedTensor<T, Dim> optimized_add(const AdvancedTensor<T, Dim>& other) const {
        AdvancedTensor<T, Dim> result(this->shape());
        optimize_add(*this, other, result);
        return result;
    }

    AdvancedTensor<T, Dim> optimized_sub(const AdvancedTensor<T, Dim>& other) const {
        AdvancedTensor<T, Dim> result(this->shape());
        optimize_sub(*this, other, result);
        return result;
    }

    AdvancedTensor<T, Dim> optimized_mul(const AdvancedTensor<T, Dim>& other) const {
        AdvancedTensor<T, Dim> result(this->shape());
        optimize_mul(*this, other, result);
        return result;
    }
    AdvancedTensor<T, Dim> optimized_div(const AdvancedTensor<T, Dim>& other) const {
        AdvancedTensor<T, Dim> result(this->shape());
        optimize_div(*this, other, result);
        return result;
    } 

protected:
    template<size_t D1, size_t D2>
    static void compute_broadcast_shape(const std::array<size_t, D1>& shape1, 
                                        const std::array<size_t, D2>& shape2, 
                                        std::array<size_t, std::max(D1, D2)>& result) {
        constexpr size_t max_dim = std::max(D1, D2);
        for (size_t i = 0; i < max_dim; ++i) {
            size_t dim1 = i < D1 ? shape1[D1 - 1 - i] : 1;
            size_t dim2 = i < D2 ? shape2[D2 - 1 - i] : 1;
            result[max_dim - 1 - i] = std::max(dim1, dim2);
        }
    }

    template<size_t D1, size_t D2, size_t ResD>
    static void broadcast_add_impl(const Tensor<T, D1>& t1, 
                                   const Tensor<T, D2>& t2, 
                                   AdvancedTensor<T, ResD>& result, 
                                   std::array<size_t, ResD> current_index, 
                                   size_t current_dim) {
        if (current_dim == ResD) {
            result(current_index) = get_value(t1, current_index, D1) + get_value(t2, current_index, D2);
            return;
        }

        for (size_t i = 0; i < result.shape()[current_dim]; ++i) {
            current_index[current_dim] = i;
            broadcast_add_impl(t1, t2, result, current_index, current_dim + 1);
        }
    }

    template<size_t D1, size_t D2>
    static T get_value(const Tensor<T, D1>& t, const std::array<size_t, D2>& index, size_t dim) {
        std::array<size_t, D1> adjusted_index;
        for (size_t i = 0; i < D1; ++i) {
            adjusted_index[i] = (i < dim) ? index[D2 - dim + i] % t.shape()[i] : 0;
        }
        return t(adjusted_index);
    }

public:
    void optimize_add(const AdvancedTensor<T, Dim>& other) {
        if (this->shape() != other.shape()) {
            throw std::invalid_argument("Tensors must have the same shape for optimize_add");
        }

        size_t size = this->data_ptr_->size();
        size_t i = 0;

        if constexpr (std::is_same<T, float>::value) {
            for (; i + 7 < size; i += 8) {
                __m256 a = _mm256_loadu_ps(&((*this->data_ptr_)[i]));
                __m256 b = _mm256_loadu_ps(&((*other.data_ptr_)[i]));
                __m256 sum = _mm256_add_ps(a, b);
                _mm256_storeu_ps(&((*this->data_ptr_)[i]), sum);
            }
        } else if constexpr (std::is_same<T, double>::value) {
            for (; i + 3 < size; i += 4) {
                __m256d a = _mm256_loadu_pd(&((*this->data_ptr_)[i]));
                __m256d b = _mm256_loadu_pd(&((*other.data_ptr_)[i]));
                __m256d sum = _mm256_add_pd(a, b);
                _mm256_storeu_pd(&((*this->data_ptr_)[i]), sum);
            }
        } else if constexpr (std::is_same<T, int>::value) {
            for (; i + 3 < size; i += 4) {
                __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&((*this->data_ptr_)[i])));
                __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&((*other.data_ptr_)[i])));
                __m128i sum = _mm_add_epi32(a, b);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&((*this->data_ptr_)[i])), sum);
            }
        }

        for (; i < size; ++i) {
            (*this->data_ptr_)[i] += (*other.data_ptr_)[i];
        }
    }

    void optimize_sub(const AdvancedTensor<T, Dim>& other) {
        if (this->shape() != other.shape()) {
            throw std::invalid_argument("Tensors must have the same shape for optimize_sub");
        }

        size_t size = this->data_ptr_->size();
        size_t i = 0;

        if constexpr (std::is_same<T, float>::value) {
            for (; i + 7 < size; i += 8) {
                __m256 a = _mm256_loadu_ps(&((*this->data_ptr_)[i]));
                __m256 b = _mm256_loadu_ps(&((*other.data_ptr_)[i]));
                __m256 diff = _mm256_sub_ps(a, b);
                _mm256_storeu_ps(&((*this->data_ptr_)[i]), diff);
            }
        } else if constexpr (std::is_same<T, double>::value) {
            for (; i + 3 < size; i += 4) {
                __m256d a = _mm256_loadu_pd(&((*this->data_ptr_)[i]));
                __m256d b = _mm256_loadu_pd(&((*other.data_ptr_)[i]));
                __m256d diff = _mm256_sub_pd(a, b);
                _mm256_storeu_pd(&((*this->data_ptr_)[i]), diff);
            }
        } else if constexpr (std::is_same<T, int>::value) {
            for (; i + 3 < size; i += 4) {
                __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&((*this->data_ptr_)[i])));
                __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&((*other.data_ptr_)[i])));
                __m128i diff = _mm_sub_epi32(a, b);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&((*this->data_ptr_)[i])), diff);
            }
        }

        for (; i < size; ++i) {
            (*this->data_ptr_)[i] -= (*other.data_ptr_)[i];
        }
    }    

    void optimize_mul(const AdvancedTensor<T, Dim>& other) {
        if (this->shape() != other.shape()) {
            throw std::invalid_argument("Tensors must have the same shape for optimize_mul");
        }

        size_t size = this->data_ptr_->size();
        size_t i = 0;

        if constexpr (std::is_same<T, float>::value) {
            for (; i + 7 < size; i += 8) {
                __m256 a = _mm256_loadu_ps(&((*this->data_ptr_)[i]));
                __m256 b = _mm256_loadu_ps(&((*other.data_ptr_)[i]));
                __m256 prod = _mm256_mul_ps(a, b);
                _mm256_storeu_ps(&((*this->data_ptr_)[i]), prod);
            }
        } else if constexpr (std::is_same<T, double>::value) {
            for (; i + 3 < size; i += 4) {
                __m256d a = _mm256_loadu_pd(&((*this->data_ptr_)[i]));
                __m256d b = _mm256_loadu_pd(&((*other.data_ptr_)[i]));
                __m256d prod = _mm256_mul_pd(a, b);
                _mm256_storeu_pd(&((*this->data_ptr_)[i]), prod);
            }
        } else if constexpr (std::is_same<T, int>::value) {
            for (; i + 3 < size; i += 4) {
                __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&((*this->data_ptr_)[i])));
                __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&((*other.data_ptr_)[i])));
                __m128i prod = _mm_mullo_epi32(a, b);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(&((*this->data_ptr_)[i])), prod);
            }
        }

        for (; i < size; ++i) {
            (*this->data_ptr_)[i] *= (*other.data_ptr_)[i];
        }
    }    

    void optimize_div(const AdvancedTensor<T, Dim>& other) {
        if (this->shape() != other.shape()) {
            throw std::invalid_argument("Tensors must have the same shape for optimize_div");
        }

        size_t size = this->data_ptr_->size();
        size_t i = 0;

        if constexpr (std::is_same<T, float>::value) {
            for (; i + 7 < size; i += 8) {
                __m256 a = _mm256_loadu_ps(&((*this->data_ptr_)[i]));
                __m256 b = _mm256_loadu_ps(&((*other.data_ptr_)[i]));
                __m256 quot = _mm256_div_ps(a, b);
                _mm256_storeu_ps(&((*this->data_ptr_)[i]), quot);
            }
        } else if constexpr (std::is_same<T, double>::value) {
            for (; i + 3 < size; i += 4) {
                __m256d a = _mm256_loadu_pd(&((*this->data_ptr_)[i]));
                __m256d b = _mm256_loadu_pd(&((*other.data_ptr_)[i]));
                __m256d quot = _mm256_div_pd(a, b);
                _mm256_storeu_pd(&((*this->data_ptr_)[i]), quot);
            }
        } else if constexpr (std::is_same<T, int>::value) {
            for (; i < size; ++i) {
                if ((*other.data_ptr_)[i] != 0) {
                    (*this->data_ptr_)[i] /= (*other.data_ptr_)[i];
                } else {
                    throw std::runtime_error("Division by zero encountered");
                }
            }
            return;
        }

        for (; i < size; ++i) {
            if ((*other.data_ptr_)[i] != 0) {
                (*this->data_ptr_)[i] /= (*other.data_ptr_)[i];
            } else {
                throw std::runtime_error("Division by zero encountered");
            }
        }
    }

};