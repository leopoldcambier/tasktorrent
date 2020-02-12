#ifndef MMIO_HPP
#define MMIO_HPP

#include <assert.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <complex>
#include <string>
#include <iomanip>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace mmio {

    template<class T> struct is_complex : std::false_type {};
    template<class T> struct is_complex<std::complex<T>> : std::true_type {};

    /** Read a real / cplx scalar **/
    template<typename V, std::enable_if_t<!is_complex<V>{}>* = nullptr>
    inline V read_entry(std::istringstream& vals) {
        V v;
        vals >> v;
        return v;
    };
    template<typename V, std::enable_if_t<is_complex<V>{}>* = nullptr>
    inline V read_line(std::istringstream& vals) {
        typename V::value_type v1, v2;
        vals >> v1 >> v2;
        return V(v1, v2);
    };

    /** Read a line real / cplx of coordinate values **/
    template<typename V, typename I_t, std::enable_if_t<!is_complex<V>{}>* = nullptr>
    inline Eigen::Triplet<V,I_t> read_line(std::istringstream& vals) {
        I_t i, j;
        V v;
        vals >> i >> j >> v;
        return Eigen::Triplet<V,I_t>(i-1,j-1,v);
    };
    template<typename V, typename I_t, std::enable_if_t<is_complex<V>{}>* = nullptr>
    inline Eigen::Triplet<V,I_t> read_line(std::istringstream& vals) {
        I_t i, j;
        typename V::value_type v1, v2;
        vals >> i >> j >> v1 >> v2;
        return Eigen::Triplet<V,I_t>(i-1,j-1,V(v1,v2));
    };

    /** Write a line real / cplx of coordinate values **/
    template<typename V, typename I_t>
    inline std::stringstream get_line(I_t i, I_t j, V v) {
        std::stringstream s;
        s << std::setprecision(20);
        s << i+1 << " " << j+1 << " " << v;
        return s;
    };
    template<typename V, typename I_t>
    inline std::stringstream get_line(I_t i, I_t j, std::complex<V> v) {
        std::stringstream s;
        s << std::setprecision(20);
        s << i+1 << " " << j+1 << " " << v.real() << " " << v.imag();
        return s;
    };    

    /** Symmetric (real) / hermitian (cplx) **/
    template<typename V, typename I_t>
    inline Eigen::Triplet<V,I_t> symmetric(Eigen::Triplet<V,I_t>& a) {
        return Eigen::Triplet<V,I_t>(a.col(), a.row(), a.value());
    }
    template<typename V, typename I_t>
    inline Eigen::Triplet<std::complex<V>,I_t> symmetric(Eigen::Triplet<std::complex<V>,I_t>& a) {
        return Eigen::Triplet<std::complex<V>,I_t>(a.col(), a.row(), std::conj(a.value()));
    }

    /** Skew-symmetric (real only, really) **/
    template<typename V, typename I_t>
    inline Eigen::Triplet<V,I_t> skew_symmetric(Eigen::Triplet<V,I_t>& a) {
        return Eigen::Triplet<V,I_t>(a.col(), a.row(), - a.value());
    }

    enum class format {coordinate, array};
    enum class type {real, integer, complex, pattern};
    enum class property {general, symmetric, hermitian, skew_symmetric};

    inline std::string prop2str(property p) {
        if(p == property::general) return "general";
        else if(p == property::symmetric) return "symmetric";
        else if(p == property::hermitian) return "hermitian";
        else return "skew_symmetric";
    }

    template<typename V>
    struct V2str {
        inline static std::string value() {
            if(is_complex<V>{}) {
                return "complex";
            } else if (std::is_integral<V>::value) {
                return "integer";
            } else {
                return "real";
            }
        }
    };

    struct Header {
        bool bannerOK;
        bool objectOK;
        format f;
        type   t;
        property p;
        Header(std::istringstream& header) {
            std::string banner, object, format, type, properties;
            header >> banner >> object >> format >> type >> properties;
            std::transform(object.begin(),      object.end(),       object.begin(),       ::tolower);
            std::transform(format.begin(),      format.end(),       format.begin(),       ::tolower);
            std::transform(type.begin(),        type.end(),         type.begin(),         ::tolower);
            std::transform(properties.begin(),  properties.end(),   properties.begin(),   ::tolower);
            this->bannerOK = ! banner.compare("%%MatrixMarket");
            this->objectOK = ! object.compare("matrix");
            assert(this->bannerOK);
            assert(this->objectOK);
            if(! format.compare("coordinate")) {
                this->f = format::coordinate;
            } else if(! format.compare("array")) {
                this->f = format::array;
            } else {
                assert(false);
            }
            if (! type.compare("real")) {
                this->t = type::real;
            } else if (! type.compare("integer")) {
                this->t = type::integer;
            } else if (! type.compare("complex")) {
                this->t = type::complex;
            } else if (! type.compare("pattern")) {
                this->t = type::pattern;
            } else { 
                assert(false);
            }
            if (! properties.compare("general")) {
                this->p = property::general;
            } else if (! properties.compare("symmetric")) {
                this->p = property::symmetric;
            } else if (! properties.compare("skew-symmetric")) {
                this->p = property::skew_symmetric;
            } else if (! properties.compare("hermitian")) {
                this->p = property::hermitian;
            } else { 
                assert(false);
            }
        }
    };

    /**
     * Read a sparse matrix in MM format
     */
    template<typename V, typename I_t>
    inline Eigen::SparseMatrix<V, Eigen::ColMajor, I_t> sp_mmread(std::string filename) {
        std::ifstream mfile(filename);
        if (mfile.is_open()) {
            std::string line;
            /** Header **/
            std::getline(mfile, line);
            std::istringstream header(line);
            Header h(header);
            assert(h.f == format::coordinate);
            assert(h.t != type::pattern);
            /** Find M N K row **/
            while(std::getline(mfile, line)) {
                if(line.size() == 0 || line[0] == '%') continue;
                else break;
            }
            I_t M, N, K;
            std::istringstream MNK(line);
            MNK >> M >> N >> K;
            std::vector<Eigen::Triplet<V,I_t>> data;
            if(h.p != property::general) {
                data.reserve(2*K);
            } else {
                data.reserve(K);
            }
            /** Read data **/
            int lineread = 0;
            while(std::getline(mfile, line)) {
                if(line.size() == 0 || line[0] == '%') continue;
                std::istringstream vals(line);
                Eigen::Triplet<V,I_t> dataline = read_line<V,I_t>(vals);
                data.push_back(dataline);
                if(dataline.row() != dataline.col() && (h.p == property::symmetric || h.p == property::hermitian)) {
                    Eigen::Triplet<V,I_t> dataline2 = symmetric(dataline);
                    data.push_back(dataline2);
                }
                if(dataline.row() != dataline.col() && (h.p == property::skew_symmetric)) {
                    Eigen::Triplet<V,I_t> dataline2 = skew_symmetric(dataline);
                    data.push_back(dataline2);
                }
                if(h.p == property::skew_symmetric && dataline.row() == dataline.col()) {
                    assert(false);
                }
                if(h.p != property::general && dataline.row() < dataline.col()) {
                    assert(false);
                }
                lineread ++;
            }
            assert(lineread == K);
            Eigen::SparseMatrix<V, Eigen::ColMajor, I_t> A(M, N);
            A.setFromTriplets(data.begin(), data.end());
            return std::move(A);
        } else {
            std::cout << "Couldn't open " << filename << std::endl;
            throw("Couldn't open file");
        }
    }

    /**
     * Reads a dense matrix in MM format
     */
    template<typename V>
    inline Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> dense_mmread(std::string filename) {
        std::ifstream mfile(filename);
        if (mfile.is_open()) {
            std::string line;
            /** Header **/
            std::getline(mfile, line);
            std::istringstream header(line);
            Header h(header);
            assert(h.p == property::general); // We don't really support anything else so far...
            assert(h.f == format::array);
            assert(h.t != type::pattern);
            /** Find M N row **/
            while(std::getline(mfile, line)) {
                if(line.size() == 0 || line[0] == '%') continue;
                else break;
            }
            int M, N;
            std::istringstream MNK(line);
            MNK >> M >> N;
            Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> A(M, N); 
            /** Read data **/
            int lineread = 0;
            while(std::getline(mfile, line)) {
                if(line.size() == 0 || line[0] == '%') continue;
                std::istringstream vals(line);
                V v = read_entry<V>(vals);
                int i = (lineread % M);
                int j = (lineread / M);
                A(i,j) = v;
                lineread ++;
            }
            assert(lineread == M*N);
            return std::move(A);
        } else {
            std::cout << "Couldn't open " << filename << std::endl;
            throw("Couldn't open file");
        }
    }

    /**
     * Writes a sparse matrix in MM format, using the optional property p.
     * Wether the matrix satisfies or not p is not verified
     */
    template<typename V, int S, typename I_t>
    inline void sp_mmwrite(std::string filename, Eigen::SparseMatrix<V,S,I_t> mat, property p = property::general) {
        std::ofstream mfile;
        mfile.open(filename);
        // mfile.open (filename);
        if (mfile.is_open()) {
            std::string type = V2str<V>::value();
            std::string prop = prop2str(p);
            mfile << "%%MatrixMarket matrix coordinate " << type << " " << prop << "\n";
            int NNZ = 0;
            for (int k = 0; k < mat.outerSize(); ++k) {
                for (typename Eigen::SparseMatrix<V,S,I_t>::InnerIterator it(mat,k); it; ++it) {
                    if( (p == property::symmetric || p == property::hermitian) && (it.row() < it.col()) ) continue;
                    if( (p == property::skew_symmetric) && (it.row() <= it.col()) ) continue;
                    NNZ ++;
                }
            }
            mfile << mat.rows() << " " << mat.cols() << " " << NNZ << "\n";
            for (int k = 0; k < mat.outerSize(); ++k) {
                for (typename Eigen::SparseMatrix<V,S,I_t>::InnerIterator it(mat,k); it; ++it) {
                    if( (p == property::symmetric || p == property::hermitian) && (it.row() < it.col()) ) continue;
                    if( (p == property::skew_symmetric) && (it.row() <= it.col()) ) continue;
                    mfile << get_line(it.row(), it.col(), it.value()).rdbuf() << "\n";
                }
            }
        } else {
            std::cout << "Couldn't open " << filename << std::endl;       
            throw("Couldn't open file");
        }
    }

    /**
     * Writes a dense matrix in MM format, using the optional property p.
     * Wether the matrix satisfies or not p is not verified
     */
    template<typename V>
    inline void dense_mmwrite(std::string filename, Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> mat, property p = property::general) {
        std::ofstream mfile;
        mfile.open (filename);
        if (mfile.is_open()) {
            std::string type = V2str<V>::value();
            std::string prop = prop2str(p);
            mfile << "%%MatrixMarket matrix array " << type << " " << prop << "\n";
            mfile << mat.rows() << " " << mat.cols() << "\n";
            for(int j = 0; j < mat.cols(); j++) {
                for(int i = 0; i < mat.rows(); i++) {
                    if( (p == property::symmetric || p == property::hermitian) && (i < j) ) continue;
                    if( (p == property::skew_symmetric) && (i <= j) ) continue;
                    mfile << mat(i,j) << "\n";
                }
            }
        } else {
            std::cout << "Couldn't open " << filename << std::endl;
            throw("Couldn't open file");
        }
    }

}

#endif