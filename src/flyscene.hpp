#ifndef __FLYSCENE__

#define __FLYSCENE__

// Must be included before glfw.
#include <GL/glew.h>

#include <GLFW/glfw3.h>
#include <tucano/effects/phongmaterialshader.hpp>
#include <tucano/mesh.hpp>
#include <tucano/shapes/camerarep.hpp>
#include <tucano/shapes/cylinder.hpp>
#include <tucano/shapes/sphere.hpp>
#include <tucano/utils/flycamera.hpp>
#include <tucano/utils/imageIO.hpp>
#include <tucano/utils/mtlIO.hpp>
#include <tucano/utils/objimporter.hpp>

const int MAX_REFLECT = 2;
const bool useSpherical = true;
const int SHADOW_SMOOTHNESS = 15;
const bool SOFT_SHADOWS = false;
const float REFLECT_BIAS = 0.005;
const int AMOUNT_FACES = 10;


class Flyscene {

public:
  Flyscene(void) {}

  /**
   * @brief Initializes the shader effect
   * @param width Window width in pixels
   * @param height Window height in pixels
   */
  void initialize(int width, int height);

  /**
   * Repaints screen buffer.
   **/
  virtual void paintGL();

  /**
   * Perform a single simulation step.
   **/
  virtual void simulate(GLFWwindow *window);

  /**
   * Returns the pointer to the flycamera instance
   * @return pointer to flycamera
   **/
  Tucano::Flycamera *getCamera(void) { return &flycamera; }

  /**
   * @brief Add a new light source
   */
  void addLight(void) { lights.push_back(flycamera.getCenter()); }

  /**
   * @brief Create a debug ray at the current camera location and passing
   * through pixel that mouse is over
   * @param mouse_pos Mouse cursor position in pixels
   */
  void createDebugRay(const Eigen::Vector2f &mouse_pos);

  /**
   * @brief raytrace your scene from current camera position   
   */
  void raytraceScene(int width = 0, int height = 0);
  /**
   * @brief trace a single ray from the camera passing through dest
   * @param origin Ray origin
   * @param dest Other point on the ray, usually screen coordinates
   * @return a RGB color
   */
Eigen::Vector3f traceRay(Eigen::Vector3f &origin, Eigen::Vector3f &dest);

  struct inters_point {
	  bool intersected;
	  Eigen::Vector3f point;
	  Tucano::Face face;
  };

  /**
   * @brief calculate intersection point
   * @param origin Ray origin
   * @param dest Other point on the ray, usually screen coordinates
   * @param normalv The vector that will be set to the normal of the corresponding face
   * @return intersection vector or origin if no intersection
   */
  Flyscene::inters_point intersection(Eigen::Vector3f origin,
	  Eigen::Vector3f dest);

  bool barycentric(Eigen::Vector3f p, std::vector<Eigen::Vector3f> vectors,
	  float& alpha, float& beta);
  /*
  * @brief calculate reflection vector
  * @param incoming Incoming ray direction
  * @param normal Normal of surface to reflect on
  * @return direction of reflection vector
  */
  Eigen::Vector3f reflect(Eigen::Vector3f incoming,
	  Eigen::Vector3f normal);

  Eigen::Vector3f shade(int level, int maxlevel, Eigen::Vector3f p,Eigen::Vector3f ray, Tucano::Face face, float shadowratio);

  Eigen::Vector3f directColor(Eigen::Vector3f p, Eigen::Vector3f ray, Tucano::Face face);

  Eigen::Vector3f reflectColor(int level, Eigen::Vector3f intersection, Eigen::Vector3f ray, Tucano::Face face);

  float shadowRatio(Eigen::Vector3f intersectionP, Tucano::Face face);
  
  //Calculates the direction of the refraction of the ray.
  Eigen::Vector3f refractionV(Eigen::Vector3f Inc, Eigen::Vector3f Outc, float r);

  Eigen::Vector3f refractColor(int level, Eigen::Vector3f intersection, Eigen::Vector3f ray, Tucano::Face face);
  

  //Calculates if the number is in range, used to check if it's in the range of frustum. (Based on c++17 function)
  float clamp(float x, float low, float high);

private:
  // A simple phong shader for rendering meshes
  Tucano::Effects::PhongMaterial phong;

  // A fly through camera
  Tucano::Flycamera flycamera;

  // the size of the image generated by ray tracing
  Eigen::Vector2i raytracing_image_size;

  // A camera representation for animating path (false means that we do not
  // render front face)
  Tucano::Shapes::CameraRep camerarep = Tucano::Shapes::CameraRep(false);

  // a frustum to represent the camera in the scene
  Tucano::Shapes::Sphere lightrep;

  // light sources for ray tracing
  vector<Eigen::Vector3f> lights;

  // Scene light represented as a camera
  Tucano::Camera scene_light;

  /// A very thin cylinder to draw a debug ray
  Tucano::Shapes::Cylinder ray = Tucano::Shapes::Cylinder(0.1, 1.0, 16, 64);

  /// A very thin cylinder to draw a debug ray
  vector<Tucano::Shapes::Cylinder> boxRays;

  ///vector containing consecutive reflections
  std::vector<Tucano::Shapes::Cylinder> reflections;

  std::vector<Tucano::Shapes::Cylinder> refractions;

  // Scene meshes
  Tucano::Mesh mesh;

  std::vector<float> makePlanes(std::vector<Tucano::Face> box);
  Eigen::Affine3f shapeModelMatrix;
  
  std::vector<std::vector<Tucano::Face>> subdivide();
  
  std::vector<std::vector<Tucano::Face>> split(std::vector<float> bounds, std::vector<Tucano::Face> bb, Eigen::Vector3f avg);

  /// MTL materials
  vector<Tucano::Material::Mtl> materials;
};

#endif // FLYSCENE
