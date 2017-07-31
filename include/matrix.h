#ifndef Matrix_h
#define Matrix_h

#include <viennacl/matrix.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/variant.hpp>

namespace heed {

    enum mode
    {
        cpu,
        gpu
    };

    template<typename T>
    class matrix {
    private:
        mode _mode;
        boost::variant<boost::numeric::ublas::matrix<T>, viennacl::matrix<T>> _data;
    public:
        matrix(mode compMode, boost::numeric::ublas::matrix<T>);
        matrix(mode compMode, viennacl::matrix<T>);
        matrix(mode compMode, std::size_t rows, std::size_t cols, std::vector<T> data);
    };
}

#endif