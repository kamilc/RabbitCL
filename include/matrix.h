#ifndef Matrix_h
#define Matrix_h

#include <cstddef>
#include <memory>
#include <boost/compute.hpp>
#include <boost/compute/algorithm.hpp>
#include "utilities.h"
#include "matrix_size.h"
#include "context_manager.h"

using namespace std;
using namespace boost;

namespace mozart
{
    template<typename T>
    class matrix
    {
    public:
        matrix();
        matrix(size_t size1, size_t size2);
        matrix(matrix source, size_t start1, size_t size1, size_t start2, size_t size2);
        matrix(std::shared_ptr<compute::vector<T>> data, size_t start1, size_t size1, size_t isize1, size_t start2, size_t size2, size_t isize2);

        static matrix view(matrix source, size_t start1, size_t end1, size_t start2, size_t end2);

        size_t size1() const;
        size_t size2() const;

        void set(size_t at1, size_t at2, T value);
        void set_data(std::vector<T>& data);
        void fill_randn(T mean, T stddev);

        matrix_size size() const;
        size_t total_size() const;
        compute::vector<T>& data() const;

        T operator()(size_t at1, size_t at2);
        size_t index(size_t at1, size_t at2) const;
    private:
        bool _transposed = false;
        size_t _size1;
        size_t _internal_size1;
        size_t _size2;
        size_t _internal_size2;
        size_t _start1;
        size_t _start2;
        std::shared_ptr<compute::vector<T>> _data;
    };

    template<typename T>
    ostream& operator<<(ostream& os, const matrix<T>& mat);

    template<typename T>
    matrix<T> dot(matrix<T>& lhs, matrix<T>& rhs);

    template<typename T>
    matrix<T> operator*(T lhs, const matrix<T>& rhs);
}

#endif