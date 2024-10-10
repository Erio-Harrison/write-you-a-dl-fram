#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>
#include "../tensor/tensor_advanced.hpp"

template<typename T, size_t Dim>
class Variable;

class Operation {
public:
    virtual ~Operation() = default;
    virtual void forward() = 0;
    virtual void backward() = 0;

    std::vector<void*> prev_variables;
};

template<typename T, size_t Dim>
class Variable {
private:
    AdvancedTensor<T, Dim> data_;
    AdvancedTensor<T, Dim> grad_;
    std::shared_ptr<Operation> grad_fn_;
    bool requires_grad_;

public:
    Variable(const AdvancedTensor<T, Dim>& data, bool requires_grad = true)
        : data_(data), requires_grad_(requires_grad) {
        if (requires_grad) {
            grad_ = AdvancedTensor<T, Dim>(data.shape());
        }
    }

    const AdvancedTensor<T, Dim>& data() const { return data_; }
    AdvancedTensor<T, Dim>& data() { return data_; }
    const AdvancedTensor<T, Dim>& grad() const { return grad_; }
    AdvancedTensor<T, Dim>& grad() { return grad_; }
    bool requires_grad() const { return requires_grad_; }
    void set_grad_fn(std::shared_ptr<Operation> grad_fn) { grad_fn_ = grad_fn; }
    std::shared_ptr<Operation> grad_fn() { return grad_fn_; }

    void backward() {
        if (!requires_grad_) {
            throw std::runtime_error("Variable does not require gradients");
        }

        if (grad_.data_ptr()->empty()) {
            grad_ = AdvancedTensor<T, Dim>(data_.shape());
        }
        
        if (grad_.data_ptr()->size() == 1) {
            (*grad_.data_ptr())[0] = 1;
        }

        std::unordered_set<Variable<T, Dim>*> visited;
        std::function<void(Variable<T, Dim>*)> backward_impl;
        backward_impl = [&](Variable<T, Dim>* v) {
            if (visited.find(v) != visited.end()) {
                return;
            }
            visited.insert(v);

            if (v->grad_fn_) {
                v->grad_fn_->backward();
                for (auto& prev : v->grad_fn_->prev_variables) {
                    backward_impl(static_cast<Variable<T, Dim>*>(prev));
                }
            }
        };

        backward_impl(this);
    }
};

template<typename T, size_t Dim>
class AddOperation : public Operation {
private:
    Variable<T, Dim>* lhs_;
    Variable<T, Dim>* rhs_;
    Variable<T, Dim>* result_;

public:
    AddOperation(Variable<T, Dim>* lhs, Variable<T, Dim>* rhs, Variable<T, Dim>* result)
        : lhs_(lhs), rhs_(rhs), result_(result) {
        this->prev_variables = {lhs, rhs};
    }

    void forward() override {
        result_->data().optimize_add(lhs_->data());
        result_->data().optimize_add(rhs_->data());
    }

    void backward() override {
        if (lhs_->requires_grad()) {
            lhs_->grad().optimize_add(result_->grad());
        }
        if (rhs_->requires_grad()) {
            rhs_->grad().optimize_add(result_->grad());
        }
    }
};

template<typename T, size_t Dim>
class SubOperation : public Operation {
private:
    Variable<T, Dim>* lhs_;
    Variable<T, Dim>* rhs_;
    Variable<T, Dim>* result_;

public:
    SubOperation(Variable<T, Dim>* lhs, Variable<T, Dim>* rhs, Variable<T, Dim>* result)
        : lhs_(lhs), rhs_(rhs), result_(result) {
        this->prev_variables = {lhs, rhs};
    }

    void forward() override {
        result_->data() = lhs_->data();
        result_->data().optimize_sub(rhs_->data());
    }

    void backward() override {
        if (lhs_->requires_grad()) {
            lhs_->grad().optimize_add(result_->grad());
        }
        if (rhs_->requires_grad()) {
            for (size_t i = 0; i < rhs_->grad().data_ptr()->size(); ++i) {
                (*rhs_->grad().data_ptr())[i] -= (*result_->grad().data_ptr())[i];
            }
        }
    }
};

template<typename T, size_t Dim>
class MulOperation : public Operation {
private:
    Variable<T, Dim>* lhs_;
    Variable<T, Dim>* rhs_;
    Variable<T, Dim>* result_;

public:
    MulOperation(Variable<T, Dim>* lhs, Variable<T, Dim>* rhs, Variable<T, Dim>* result)
        : lhs_(lhs), rhs_(rhs), result_(result) {
        this->prev_variables = {lhs, rhs};
    }

    void forward() override {
        result_->data() = lhs_->data();
        result_->data().optimize_mul(rhs_->data());
    }

    void backward() override {
        if (lhs_->requires_grad()) {
            AdvancedTensor<T, Dim> temp = result_->grad();
            temp.optimize_mul(rhs_->data());
            lhs_->grad().optimize_add(temp);
        }
        if (rhs_->requires_grad()) {
            AdvancedTensor<T, Dim> temp = result_->grad();
            temp.optimize_mul(lhs_->data());
            rhs_->grad().optimize_add(temp);
        }
    }

};

template<typename T, size_t Dim>
class DivOperation : public Operation {
private:
    Variable<T, Dim>* lhs_;
    Variable<T, Dim>* rhs_;
    Variable<T, Dim>* result_;

public:
    DivOperation(Variable<T, Dim>* lhs, Variable<T, Dim>* rhs, Variable<T, Dim>* result)
        : lhs_(lhs), rhs_(rhs), result_(result) {
        this->prev_variables = {lhs, rhs};
    }

    void forward() override {
        result_->data() = lhs_->data();
        result_->data().optimize_div(rhs_->data());
    }

    void backward() override {
        if (lhs_->requires_grad()) {
            AdvancedTensor<T, Dim> temp = result_->grad();
            temp.optimize_div(rhs_->data());
            lhs_->grad().optimize_add(temp);
        }
        if (rhs_->requires_grad()) {
            AdvancedTensor<T, Dim> temp = result_->grad();
            temp.optimize_mul(result_->data());
            temp.optimize_div(rhs_->data());
            for (size_t i = 0; i < rhs_->grad().data_ptr()->size(); ++i) {
                (*rhs_->grad().data_ptr())[i] -= (*temp.data_ptr())[i];
            }
        }
    }
};

template<typename T, size_t Dim>
Variable<T, Dim> operator+(Variable<T, Dim>& lhs, Variable<T, Dim>& rhs) {
    AdvancedTensor<T, Dim> result_data(lhs.data().shape());
    result_data = lhs.data();
    result_data.optimize_add(rhs.data());
    Variable<T, Dim> result(result_data, lhs.requires_grad() || rhs.requires_grad());
    if (result.requires_grad()) {
        auto op = std::make_shared<AddOperation<T, Dim>>(&lhs, &rhs, &result);
        result.set_grad_fn(op);
    }
    return result;
}

template<typename T, size_t Dim>
Variable<T, Dim> operator-(Variable<T, Dim>& lhs, Variable<T, Dim>& rhs) {
    AdvancedTensor<T, Dim> result_data(lhs.data().shape());
    result_data = lhs.data();
    result_data.optimize_sub(rhs.data());
    Variable<T, Dim> result(result_data, lhs.requires_grad() || rhs.requires_grad());
    if (result.requires_grad()) {
        auto op = std::make_shared<SubOperation<T, Dim>>(&lhs, &rhs, &result);
        result.set_grad_fn(op);
    }
    return result;
}

template<typename T, size_t Dim>
Variable<T, Dim> operator*(Variable<T, Dim>& lhs, Variable<T, Dim>& rhs) {
    AdvancedTensor<T, Dim> result_data(lhs.data().shape());
    result_data = lhs.data();
    result_data.optimize_mul(rhs.data());
    Variable<T, Dim> result(result_data, lhs.requires_grad() || rhs.requires_grad());
    if (result.requires_grad()) {
        auto op = std::make_shared<MulOperation<T, Dim>>(&lhs, &rhs, &result);
        result.set_grad_fn(op);
    }
    return result;
}

template<typename T, size_t Dim>
Variable<T, Dim> operator/(Variable<T, Dim>& lhs, Variable<T, Dim>& rhs) {
    AdvancedTensor<T, Dim> result_data(lhs.data().shape());
    result_data = lhs.data();
    result_data.optimize_div(rhs.data());
    Variable<T, Dim> result(result_data, lhs.requires_grad() || rhs.requires_grad());
    if (result.requires_grad()) {
        auto op = std::make_shared<DivOperation<T, Dim>>(&lhs, &rhs, &result);
        result.set_grad_fn(op);
    }
    return result;
}