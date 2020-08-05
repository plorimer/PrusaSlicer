#ifndef SLIC3R_TEST_UTILS
#define SLIC3R_TEST_UTILS

#include <libslic3r/TriangleMesh.hpp>
#include <libslic3r/Format/OBJ.hpp>
#include <libslic3r/Format/STL.hpp>

#if defined(WIN32) || defined(_WIN32)
#define PATH_SEPARATOR R"(\)"
#else
#define PATH_SEPARATOR R"(/)"
#endif

inline Slic3r::TriangleMesh load_model(const std::string &obj_filename)
{
    Slic3r::TriangleMesh mesh;
    auto fpath = TEST_DATA_DIR PATH_SEPARATOR + obj_filename;

    std::string ext; ext.reserve(4);
    auto it = obj_filename.rbegin();
    while (it != obj_filename.rend() && *it != '.')
        ext.append(1, std::tolower(*it++));

    std::reverse(ext.begin(), ext.end());

    if (ext == "obj")
        Slic3r::load_obj(fpath.c_str(), &mesh);
    else if (ext == "stl")
        Slic3r::load_stl(fpath.c_str(), &mesh);

    return mesh;
}

#endif // SLIC3R_TEST_UTILS
