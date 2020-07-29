#include <catch2/catch.hpp>

#include <libslic3r/ExPolygon.hpp>
#include <libslic3r/BoundingBox.hpp>
#include <libslic3r/SLA/AGGRaster.hpp>

#include "sla_test_utils.hpp"

namespace Slic3r { namespace sla {

using FloatRaster = AGGRaster<agg::pixfmt_alpha_blend_gray<agg::blender_gray32, agg::row_accessor<float>>,
                              agg::renderer_scanline_aa_solid,
                              agg::rasterizer_scanline_aa<>,
                              agg::scanline_p8>;

struct PPMFloatRasterEncoder {
    EncodedRaster operator()(const void *ptr,
                             size_t      w,
                             size_t      h,
                             size_t      num_components)
    {
        std::vector<uint8_t> buf;

        auto header = std::string("P5 ") +
                      std::to_string(w) + " " +
                      std::to_string(h) + " " + "255 ";

        auto sz = w * h * num_components;
        size_t s = sz + header.size();

        buf.reserve(s);

        std::copy(header.begin(), header.end(), std::back_inserter(buf));

        for (size_t i = 0; i < sz; ++i)
            buf.emplace_back(
                uint8_t(std::max(0., std::min(255., 255. * static_cast<const float *>(ptr)[i])))
            );

        return EncodedRaster(std::move(buf), "ppm");
    }
};

class PressureRaster: public FloatRaster
{

    template<class Op>
    PressureRaster & apply_op(const PressureRaster &other, Op &&op)
        noexcept(noexcept(op(float{}, float{})))
    {
        assert(resolution() == other.resolution());

        for (size_t i = 0; i < m_buf.size(); ++i)
            m_buf[i] = op(m_buf[i], other.m_buf[i]);

        return *this;
    }

public:
    PressureRaster(const RasterBase::Resolution &res,
                   const RasterBase::PixelDim &  pxdim)
        : FloatRaster{res,
               pxdim,
               RasterBase::Trafo{},
               FloatRaster::TColor{0.f},
               FloatRaster::TColor{1.f},
               agg::gamma_power{1.}}
    {}

    inline const float& operator()(size_t col, size_t row) const noexcept
    {
        return m_buf[row * resolution().width_px + col];
    }

    inline float& operator()(size_t col, size_t row) noexcept
    {
        return m_buf[row * resolution().width_px + col];
    }

    inline PressureRaster & operator+=(float val) noexcept
    {
        for (size_t i = 0; i < m_buf.size(); ++i) m_buf[i] += val;
        return *this;
    }

    inline PressureRaster & operator+=(const PressureRaster &other) noexcept
    {
        return apply_op(other, std::plus<float>{});
    }

    inline PressureRaster & operator*=(const PressureRaster &other) noexcept
    {
        return apply_op(other, std::multiplies<float>{});
    }

    const std::vector<float> & data() { return m_buf; }

};

class PressureRaster8 : public RasterGrayscaleAAGammaPower {
    inline uint8_t &get(size_t c, size_t r)
    {
        return m_buf[r * m_resolution.width_px + c];
    }

public:

    using RasterGrayscaleAAGammaPower::RasterGrayscaleAAGammaPower;

    void swap(std::vector<uint8_t> &buf) { m_buf.swap(buf); }

    void blur(double /*dist*/)
    {
        for (size_t r = 1; r < m_resolution.height_px - 1; ++r)
            for (size_t c = 1; c < m_resolution.width_px - 1; ++c) {
                auto v =
                    (double(get(c - 1, r + 1)) + get(c, r + 1) + get(c + 1, r + 1) +
                     get(c - 1, r) + get(c, r) + get(c + 1, r) +
                     get(c - 1, r - 1) + get(c, r - 1) + get(c + 1, r - 1)) / 9.;
                get(c, r) = uint8_t(std::round(v));
            }
    }
};

double bb_width(const BoundingBoxf &bb) { return bb.max.x() - bb.min.x(); }
double bb_height(const BoundingBoxf &bb) { return bb.max.y() - bb.min.y(); }

// Help cache coherency by having the same memory layout as the raster
using PressureMatrix = Eigen::MatrixXf;

EncodedRaster to_ppm(const PressureMatrix &pmat, float cmin, float cmax)
{
    std::vector<uint8_t> buf;

    auto header = std::string("P5 ") +
                  std::to_string(pmat.rows()) + " " +
                  std::to_string(pmat.cols()) + " " + "255 ";

    auto sz = size_t(pmat.cols() * pmat.rows());
    size_t s = sz + header.size();

    buf.reserve(s);

    std::copy(header.begin(), header.end(), std::back_inserter(buf));

    for (size_t r = 0; r < size_t(pmat.cols()); ++r)
        for (size_t c = 0; c < size_t(pmat.rows()); ++c)
            buf.emplace_back(
                uint8_t(std::max(0., std::min(255., 255. * (pmat(c, r) - cmin) / (cmax - cmin) )))
                );

    return EncodedRaster(std::move(buf), "ppm");
}

EncodedRaster to_ppm(const PressureMatrix &pmat)
{
    return to_ppm(pmat, pmat.minCoeff(), pmat.maxCoeff());
}

class PressureModel {
    const std::vector<ExPolygons> &m_slices;
    const std::vector<float>      &m_heights;

    RasterBase::Resolution m_res;
    RasterBase::PixelDim   m_pxdim;
    Point m_bedcenter = {0, 0};

    float m_tear_foce = 1e-2f; // g / mm3
    float m_material_density = 1e-3f; // g / mm3
    float m_bed_stick        = 150.f * m_material_density + 2 * m_tear_foce;

    void init_grid(PressureMatrix &rst) const
    {
        rst.setOnes();
        rst *= m_bed_stick * area(m_pxdim); // TODO: substitue grip force of the print bed
    }

    void calc_grid(size_t n, PressureMatrix &pmat, std::vector<uint8_t> &prev_rst) const
    {
        PressureRaster8 rst(m_res, m_pxdim,
                            RasterBase::Trafo{}.set_center(m_bedcenter));

        for (auto &expoly : m_slices[n]) rst.draw(expoly);
        rst.blur(0.);

        float h = n == 0 || m_heights.empty() ? 0. : m_heights[n] - m_heights[n - 1];

        float pxforce = area(m_pxdim) * m_tear_foce;
        float weight = h * area(m_pxdim) * m_material_density;

        for (int y = 0; y < int(m_res.height_px); ++y) {
            for (int x = 0; x < int(m_res.width_px); ++x) {
                float v = pmat(x, y);
                float mask = rst.read_pixel(x, y) / 255.;
                float prevmask = !std::signbit(v) * prev_rst[y * m_res.width_px + x] / 255.;

                pmat(x, y) = v = mask * (v - pxforce - weight) + prevmask * pxforce;

//                m_minval = std::min(m_minval, v);
//                m_maxval = std::max(m_maxval, v);
            }
        }

        rst.swap(prev_rst);
    }

public:

    mutable float m_minval = std::numeric_limits<float>::max();
    mutable float m_maxval = std::numeric_limits<float>::lowest();

    PressureModel(const std::vector<ExPolygons> &slices,
                  const std::vector<float> &     heights,
                  const BoundingBoxf &            bed,
                  const Vec<2, size_t> & gridsize)
        : m_slices{slices}
        , m_heights{heights}
        , m_res{gridsize.x(), gridsize.y()}
        , m_pxdim{bb_width(bed) / gridsize.x(), bb_height(bed) / gridsize.y()}
        , m_bedcenter{scaled(bed.center())} {};

    std::vector<PressureMatrix> operator() (size_t n) const;
};

std::vector<PressureMatrix> PressureModel::operator() (size_t n) const
{
    std::vector<PressureMatrix> ret; ret.reserve(n);

    PressureMatrix grid{m_res.width_px, m_res.height_px};
    std::vector<uint8_t> prev_rst(m_res.pixels(), 0);

    init_grid(grid);

    for (size_t i = 0; i < n && i < m_slices.size(); ++i) {
        calc_grid(i, grid, prev_rst);
        ret.emplace_back(grid);
    }

    m_minval = -(n/2 * area(m_pxdim) * m_tear_foce);
    m_maxval = 0.; // m_bed_stick * area(m_pxdim);

    return ret;
}


TEST_CASE("PressureMatrix calc", "[SupGen]")
{

    TriangleMesh mesh = make_pyramid(20., 8.);
    mesh.rotate_y(PI);
    mesh.translate(0.f, 0.f, 5.f);


//    TriangleMesh mesh = make_cube(25., 25., 150.);
////    mesh.rotate_x(PI/4);
////    mesh.rotate_y(PI/4);
////    mesh.rotate_z(PI/4);
//    mesh.translate(-mesh.bounding_box().center().cast<float>());
//    mesh.require_shared_vertices();
//    mesh.WriteOBJFile("rotcube.obj");

    // Prepare the slice grid and the slices
    std::vector<ExPolygons> slices;
    auto                    bb      = cast<float>(mesh.bounding_box());
    std::vector<float>      heights = grid(bb.min.z() - 0.1f, bb.max.z(), 0.1f);
    slice_mesh(mesh, heights, slices, CLOSING_RADIUS, [] {});

    PressureModel pm{slices, heights, BoundingBoxf{{0., 0.}, {120.f, 68.f}}, {1200, 680}};

    std::vector<PressureMatrix> pmats = pm(slices.size());

    size_t n = 0;
    for (const PressureMatrix &pmat : pmats) {
        std::fstream{std::string("pmat") + std::to_string(n++) + ".ppm", std::fstream::out} << to_ppm(pmat, pm.m_minval, pm.m_maxval);
    }
}

}} // namespace Slic3r::sla
