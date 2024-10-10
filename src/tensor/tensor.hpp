#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <memory>
#include <initializer_list>
#include <fstream>

template<typename T, size_t Dim>
class Tensor {
public:
    Tensor() : shape_(), data_ptr_(std::make_shared<std::vector<T>>()) {}

protected:
    std::shared_ptr<std::vector<T>> data_ptr_;

private:
    std::array<size_t, Dim> shape_;

    size_t get_flat_index(const std::array<size_t, Dim>& indices) const {
        size_t flat_index = 0;
        size_t multiplier = 1;
        for (int i = Dim - 1; i >= 0; --i) {
            flat_index += indices[i] * multiplier;
            multiplier *= shape_[i];
        }
        return flat_index;
    }

public:
    Tensor(const std::array<size_t, Dim>& shape) : shape_(shape) {
        size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        data_ptr_ = std::make_shared<std::vector<T>>(total_size);
    }

    Tensor(const std::array<size_t, Dim>& shape, const std::vector<T>& data) : shape_(shape), data_ptr_(std::make_shared<std::vector<T>>(data)) {
        if (data.size() != std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())) {
            throw std::invalid_argument("Data size does not match shape");
        }
    }

    Tensor<T, 2> matmul(const Tensor<T, 2>& other) const {
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Invalid dimensions for matrix multiplication");
        }
        Tensor<T, 2> result({shape_[0], other.shape_[1]});
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < other.shape_[1]; ++j) {
                T sum = 0;
                for (size_t k = 0; k < shape_[1]; ++k) {
                    sum += (*this)({i, k}) * other({k, j});
                }
                result({i, j}) = sum;
            }
        }
        return result;
    }

    Tensor<T, 2> transpose() const {
        Tensor<T, 2> result({shape_[1], shape_[0]});
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result({j, i}) = (*this)({i, j});
            }
        }
        return result;
    }

    T& operator()(const std::array<size_t, Dim>& indices) {
        return (*data_ptr_)[get_flat_index(indices)];
    }

    const T& operator()(const std::array<size_t, Dim>& indices) const {
        return (*data_ptr_)[get_flat_index(indices)];
    }

    const std::array<size_t, Dim>& shape() const {
        return shape_;
    }

    const std::shared_ptr<std::vector<T>>& data_ptr() const { return data_ptr_; }

    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Unable to open file for writing");
        }

        file.write(reinterpret_cast<const char*>(shape_.data()), sizeof(size_t) * Dim);
        file.write(reinterpret_cast<const char*>(data_ptr_->data()), sizeof(T) * data_ptr_->size());
    }

    static Tensor<T, Dim> load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Unable to open file for reading");
        }

        std::array<size_t, Dim> shape;
        file.read(reinterpret_cast<char*>(shape.data()), sizeof(size_t) * Dim);

        Tensor<T, Dim> result(shape);
        file.read(reinterpret_cast<char*>(result.data_ptr_->data()), sizeof(T) * result.data_ptr_->size());

        return result;
    }

    Tensor<T, Dim> operator+(const Tensor<T, Dim>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
        Tensor<T, Dim> result(shape_);
        std::transform(data_ptr_->begin(), data_ptr_->end(), other.data_ptr_->begin(), result.data_ptr_->begin(), std::plus<T>());
        return result;
    }

    Tensor<T, Dim> operator-(const Tensor<T, Dim>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
        Tensor<T, Dim> result(shape_);
        std::transform(data_ptr_->begin(), data_ptr_->end(), other.data_ptr_->begin(), result.data_ptr_->begin(), std::minus<T>());
        return result;
    }

    Tensor<T, Dim> operator*(const Tensor<T, Dim>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
        Tensor<T, Dim> result(shape_);
        std::transform(data_ptr_->begin(), data_ptr_->end(), other.data_ptr_->begin(), result.data_ptr_->begin(), std::multiplies<T>());
        return result;
    }

    Tensor<T, Dim> operator/(const Tensor<T, Dim>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes do not match");
        }
        Tensor<T, Dim> result(shape_);
        std::transform(data_ptr_->begin(), data_ptr_->end(), other.data_ptr_->begin(), result.data_ptr_->begin(), std::divides<T>());
        return result;
    }

};
