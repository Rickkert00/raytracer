#include "flyscene.hpp"
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <float.h>
#include <Tucano/shapes/box.hpp>
#include <thread>
#include <future>


//value used to check 
constexpr float minCheck = 1e-8;
const Eigen::Vector4f backgroundColor = Eigen::Vector4f(0.9, 0.9, 0.9, 0);
std::vector<std::vector<Tucano::Face>> bboxes;
vector<vector<Eigen::Vector3f>> pixel_data;
double shadowBias = 1e-4;
double reflectBias = 1e-2;
int pixel = 0;
int percentage = 0;
std::vector<Tucano::Shapes::Box> allBoxes;

void Flyscene::initialize(int width, int height) {
	// initiliaze the Phong Shading effect for the Opengl Previewer
	phong.initialize();

	// set the camera's projection matrix
	flycamera.setPerspectiveMatrix(60.0, width / (float)height, 0.1f, 100.0f);
	flycamera.setViewport(Eigen::Vector2f((float)width, (float)height));

	// load the OBJ file and materials
	Tucano::MeshImporter::loadObjFile(mesh, materials,
		"resources/models/scene.obj");

	// normalize the model (scale to unit cube and center at origin)
	mesh.normalizeModelMatrix();
	shapeModelMatrix = mesh.getShapeModelMatrix();

	// pass all the materials to the Phong Shader
	for (int i = 0; i < materials.size(); ++i)
		phong.addMaterial(materials[i]);

	// set the color and size of the sphere to represent the light sources
	// same sphere is used for all sources
	lightrep.setColor(Eigen::Vector4f(1.0, 1.0, 0.0, 1.0));
	lightrep.setSize(0.15);

	// create a first ray-tracing light source at some random position
	//lights.push_back(Eigen::Vector3f(-1.0, 1.0, 1.0));

	bboxes = subdivide();
	int sum = 0;
	int numberOfFaces = mesh.getNumberOfFaces();
	
	cout << "number of boxes: "<< bboxes.size()<<endl;
	//for (int i = 0; i < bboxes.size(); i++) {
	//	std::vector<Tucano::Face> box = bboxes[i];
	//	cout << "BOX\n";
	//	vector<float> myCoords = makePlanes(box);
	//	cout << "COORDS\n" << myCoords[0] << " " << myCoords[1] << " " << myCoords[2] << " " << myCoords[3] << " " << myCoords[4] << " " << myCoords[5] << endl;

	//	float xdiff = myCoords[0] - myCoords[3];
	//	xdiff = xdiff < 0 ? -xdiff : xdiff;
	//	float ydiff = myCoords[1] - myCoords[4];
	//	ydiff = ydiff < 0 ? -ydiff : ydiff;
	//	float zdiff = myCoords[2] - myCoords[5];
	//	zdiff = zdiff < 0 ? -zdiff : zdiff;

	//	cout << "DIFF\n" << xdiff << " " << ydiff << " " << zdiff << endl;

	//	for (int j = 0; j < box.size(); j++) {
	//		Tucano::Face face = box[j];

	//		Eigen::Vector4f homogeneous0 = shapeModelMatrix * mesh.getVertex(face.vertex_ids[0]);
	//		Eigen::Vector4f v0 = Eigen::Vector4f(homogeneous0.x() / homogeneous0.w(), homogeneous0.y() / homogeneous0.w(), homogeneous0.z() / homogeneous0.w(), 1.0);
	//		Eigen::Vector4f homogeneous1 = shapeModelMatrix * mesh.getVertex(face.vertex_ids[1]);
	//		Eigen::Vector4f v1 = Eigen::Vector4f(homogeneous1.x() / homogeneous1.w(), homogeneous1.y() / homogeneous1.w(), homogeneous1.z() / homogeneous1.w(), 1.0);
	//		Eigen::Vector4f homogeneous2 = shapeModelMatrix * mesh.getVertex(face.vertex_ids[2]);
	//		Eigen::Vector4f v2 = Eigen::Vector4f(homogeneous2.x() / homogeneous2.w(), homogeneous2.y() / homogeneous2.w(), homogeneous2.z() / homogeneous2.w(), 1.0);

	//		cout << "FACE\n";
	//		cout << v0.x() << " " << v0.y() << " " << v0.z() << endl;
	//		cout << v1.x() << " " << v1.y() << " " << v1.z() << endl;
	//		cout << v2.x() << " " << v2.y() << " " << v2.z() << endl;
	//	}
	//	sum += box.size();
	//	cout << box.size() << endl;
	//}
	//cout << sum << " " << numberOfFaces;
	
	// scale the camera representation (frustum) for the ray debug
	camerarep.shapeMatrix()->scale(0.2);

	// the debug ray is a cylinder, set the radius and length of the cylinder
	ray.setSize(0.005, 10.0);

	// craete a first debug ray pointing at the center of the screen
	createDebugRay(Eigen::Vector2f(width / 2.0, height / 2.0));

	//for (int i = 0; i < bboxes.size(); i++) {
	//	std::vector<float> bounds = makePlanes(bboxes[i]);
	//	float xdiff = bounds[0] - bounds[3];
	//	xdiff = xdiff < 0 ? -xdiff : xdiff;
	//	float ydiff = bounds[1] - bounds[4];
	//	ydiff = ydiff < 0 ? -ydiff : ydiff;
	//	float zdiff = bounds[2] - bounds[5];
	//	zdiff = zdiff < 0 ? -zdiff : zdiff;

	//	Tucano::Shapes::Cylinder boxRay = Tucano::Shapes::Cylinder(0.1, 1.0, 16, 64);
	//	boxRay.setOriginOrientation(Eigen::Vector3f(bounds[0], bounds[1], bounds[2]), Eigen::Vector3f(bounds[3], bounds[4], bounds[5]));
	//	boxRay.setSize(0.005, pow((pow(xdiff, 2) + pow(ydiff, 2) + pow(zdiff, 2)), 0.5));
	//	boxRays.push_back(boxRay);
	//}
	glEnable(GL_DEPTH_TEST);
}

void Flyscene::paintGL(void) {

	// update the camera view matrix with the last mouse interactions
	flycamera.updateViewMatrix();
	Eigen::Vector4f viewport = flycamera.getViewport();

	// clear the screen and set background color
	glClearColor(backgroundColor.x(), backgroundColor.y(), backgroundColor.z(), backgroundColor.w());
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// position the scene light at the last ray-tracing light source
	scene_light.resetViewMatrix();
	if (lights.size() != 0 ) {
		scene_light.viewMatrix()->translate(-lights.back());
	}
	
	for (int i = 0; i < allBoxes.size(); ++i) {
		allBoxes[i].render(flycamera, scene_light);
	}

	//for (int i = 0; i < 100; i++)
		//cout << mesh.getVertexColor(0);

	// render the scene using OpenGL and one light source
	phong.render(mesh, flycamera, scene_light);

	// render the ray and camera representation for ray debug
	ray.render(flycamera, scene_light);

	// render bb cylinders
	//for (int i = 0; i < boxRays.size(); i++) {
	//	boxRays[i].render(flycamera, scene_light);
	//}

	camerarep.render(flycamera, scene_light);

	//render reflections
	for (int i = 0; i < reflections.size(); i++) {
		reflections[i].render(flycamera, scene_light);
	}

	for (int i = 0; i < refractions.size(); i++) {
		refractions[i].render(flycamera, scene_light);
	}

	// render ray tracing light sources as yellow spheres
	for (int i = 0; i < lights.size(); ++i) {
		lightrep.resetModelMatrix();
		lightrep.modelMatrix()->translate(lights[i]);
		lightrep.render(flycamera, scene_light);
	}

	// render coordinate system at lower right corner
	flycamera.renderAtCorner();
}

void Flyscene::simulate(GLFWwindow* window) {
	// Update the camera.
	// NOTE(mickvangelderen): GLFW 3.2 has a problem on ubuntu where some key
	// events are repeated: https://github.com/glfw/glfw/issues/747. Sucks.
	float dx = (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS ? 0.5 : 0.0) -
		(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS ? 0.5 : 0.0);
	float dy = (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS ||
		glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS
		? 0.5
		: 0.0) -
		(glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS ||
			glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS
			? 0.5
			: 0.0);
	float dz = (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS ? 0.5 : 0.0) -
		(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS ? 0.5 : 0.0);
	flycamera.translate(dx, dy, dz);
}

void Flyscene::createDebugRay(const Eigen::Vector2f& mouse_pos) {
	ray.resetModelMatrix();
	// from pixel position to world coordinates
	Eigen::Vector3f screen_pos = flycamera.screenToWorld(mouse_pos);

	// direction from camera center to click position
	Eigen::Vector3f dir = (screen_pos - flycamera.getCenter()).normalized();
	Eigen::Vector3f origin = flycamera.getCenter();

	//calculate intersection point with scene(closest intersection found)
	inters_point intersectionstruc = intersection(origin, screen_pos);

	Eigen::Vector3f shadingresult = shade(0, MAX_REFLECT, intersectionstruc.point, intersectionstruc.point - origin, intersectionstruc.face, 1);
	std::cout << "color: " << shadingresult << std::endl;

	//if intersection is the infinite vector, the ray intersects with no triangle
	ray.setOriginOrientation(flycamera.getCenter(), dir);
	if (intersectionstruc.intersected) {
		float height = (intersectionstruc.point - flycamera.getCenter()).norm();
		ray.setSize(0.005, height);
	}
	else {
		ray.setSize(0.005, 10);
	}

	//calculate reflection ray and draw it
	//reset reflections again
	reflections.clear();
	Eigen::Vector3f previousIntersect = flycamera.getCenter();
	Eigen::Vector3f incoming = (intersectionstruc.point - previousIntersect);
	Eigen::Vector3f intersectionp = intersectionstruc.point;
	Eigen::Vector3f normalized_normal = intersectionstruc.face.normal.normalized();
	Eigen::Vector3f reflection = reflect(incoming, normalized_normal).normalized();

	// calc refractions
	refractions.clear();
	//Eigen::Vector3f refraction = refractionV(incoming, normalized_normal, materials[intersectionstruc.face.material_id].getOpticalDensity()).normalized();
	Eigen::Vector3f refraction = refractionV(incoming, normalized_normal, 1.4).normalized();

	Eigen::Vector3f incoming2 = (intersectionstruc.point - previousIntersect);
	Eigen::Vector3f intersectionp2 = intersectionstruc.point;
	Eigen::Vector3f normalized_normal2 = intersectionstruc.face.normal.normalized();
	inters_point refractionstruc = intersectionstruc;
	Eigen::Vector3f previousIntersect2 = flycamera.getCenter();
	//for (int i = 0; i < MAX_REFLECT; i++) {
	//	if (intersectionstruc.intersected) {
	//		incoming = (intersectionstruc.point - previousIntersect).normalized();
	//		normalized_normal = intersectionstruc.face.normal.normalized();
	//		intersectionp = intersectionstruc.point;

	//		reflection = reflect(incoming, normalized_normal).normalized();

	//		Tucano::Shapes::Cylinder reflected = Tucano::Shapes::Cylinder(0.005, 1.0, 16, 64);
	//		reflected.resetModelMatrix();
	//		reflected.resetShapeMatrix();
	//		reflected.resetLocations();
	//		reflected.reset();
	//		reflected.setOriginOrientation(intersectionp, reflection);

	//		intersectionstruc = intersection(intersectionstruc.point, intersectionstruc.point + reflection);
	//		if (intersectionstruc.intersected) {
	//			reflected.setSize(0.005, (intersectionstruc.point - intersectionp).norm());
	//			reflections.push_back(reflected);
	//			previousIntersect = intersectionp;
	//		}
	//		else {
	//			reflected.setSize(0.005, 10);
	//			reflections.push_back(reflected);
	//			break;
	//		}

	////insert reflection rays
	////std::cout << shadowRatio(intersectionstruc.point, intersectionstruc.face) << std::endl;
	//	}
	//}
	//for (int i = 0; i < MAX_REFLECT; i++) {
	//	if (refractionstruc.intersected) {
	//		incoming2 = (refractionstruc.point - previousIntersect2).normalized();
	//		normalized_normal2 = refractionstruc.face.normal.normalized();
	//		intersectionp2 = refractionstruc.point;
	//		refraction = refractionV(incoming2, normalized_normal2, 1.4).normalized();
	//		Tucano::Shapes::Cylinder refracted = Tucano::Shapes::Cylinder(0.005, 1.0, 16, 64);
	//		refracted.resetModelMatrix();
	//		refracted.resetShapeMatrix();
	//		refracted.resetLocations();
	//		refracted.reset();
	//		refracted.setOriginOrientation(intersectionp2, refraction);


	//		refractionstruc = intersection(refractionstruc.point, refractionstruc.point + refraction);
	//		if (refractionstruc.intersected) {
	//			refracted.setSize(0.005, (refractionstruc.point - intersectionp2).norm());
	//			//std::cout << refraction << std::endl;
	//			refractions.push_back(refracted);
	//			previousIntersect2 = intersectionp2;
	//		}
	//		else {
	//			//std::cout << refraction << std::endl;
	//			refracted.setSize(0.005, 10);
	//			refractions.push_back(refracted);
	//			break;
	//		}
	//	}
	//}

	// place the camera representation (frustum) on current camera location, 
	camerarep.resetModelMatrix();
	camerarep.setModelMatrix(flycamera.getViewMatrix().inverse());
}

void Flyscene::raytraceScene(int width, int height) {
	std::cout << "ray tracing ..." << std::endl;

	// if no width or height passed, use dimensions of current viewport
	Eigen::Vector2i image_size(width, height);
	if (width == 0 || height == 0) {
		image_size = flycamera.getViewportSize();
	}

	// create 2d vector to hold pixel colors and resize to match image size
	//vector<vector<Eigen::Vector3f>> pixel_data;
	pixel_data.resize(image_size[1]);
	for (int i = 0; i < image_size[1]; ++i) {
		pixel_data[i].resize(image_size[0]);
	}
	// origin of the ray is always the camera center
	Eigen::Vector3f origin = flycamera.getCenter();
	Eigen::Vector3f screen_coords;

	//code credit goes to https://medium.com/@phostershop/solving-multithreaded-raytracing-issues-with-c-11-7f018ecd76fa
	std::size_t max = double(image_size[0]) * double(image_size[1]);
	std::size_t cores = std::thread::hardware_concurrency();
	volatile std::atomic<std::size_t> count(0);
	std::vector<std::future<void>> future_vector;
	for (std::size_t i(0); i < cores; ++i) {
		future_vector.emplace_back(std::async([=, &origin]()
			{
				for (std::size_t index(i); index < max; index += cores)
				{
					std::size_t x = index % image_size[0];
					std::size_t y = index / image_size[0];
					Eigen::Vector3f screen_coords = flycamera.screenToWorld(Eigen::Vector2f(x, y));
					pixel_data[index % image_size[1]][index / image_size[1]] = traceRay(origin, screen_coords);
				}
			}));
	}
	//wait for threads to finish
	for (int i = 0; i < future_vector.size(); ++i) {
		if (!(future_vector[i]._Is_ready())) {
			i = -1;
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
	}
	// write the ray tracing result to a PPM image
	Tucano::ImageImporter::writePPMImage("result.ppm", pixel_data);

	std::cout << "ray tracing done! " << std::endl;
	pixel_data.clear();
}

Eigen::Vector3f  Flyscene::traceRay(Eigen::Vector3f& origin,
	Eigen::Vector3f& dest) {

	inters_point intersectionstruc = intersection(origin, dest);

	if (percentage != pixel / 10000) {
		percentage = pixel / 10000;
		std::cout << percentage << "%" << std::endl;
	}
	pixel++;

	if (intersectionstruc.intersected == true) {
		//Multiply the rgb value of the pixel by the shadow ratio
		float shadowratio = shadowRatio(intersectionstruc.point, intersectionstruc.face);
		return shade(0, MAX_REFLECT, intersectionstruc.point, intersectionstruc.point - origin, intersectionstruc.face, shadowratio);

	}
	//if miss then return background color
	return Eigen::Vector3f(backgroundColor.x(), backgroundColor.y(), backgroundColor.z());
}

//Calculates the direction of the refraction when the ray is inside the object and outside.
Eigen::Vector3f Flyscene::refractionV(Eigen::Vector3f view, Eigen::Vector3f normal, float index) {

	float cos = clamp(view.dot(normal), -1.0f, 1.0f);
	//float cos = view.normalized().dot(normal.normalized());
	float i = 1;
	float x = index;
	Eigen::Vector3f norm = normal;
	if (cos < 0) {
		cos = -cos;
	}
	else {
		//std::swap(i, x);
		norm = -normal;
	}

	float y = i / x;
	//std::cout << "y: " << y << std::endl;
	float z = 1 - (y * y) * (1 - cos * cos);
	//std::cout << "z: " << z << std::endl;

	if (z < 0) {
		return Eigen::Vector3f(0.0, 0.0, 0.0);
	}
	else {
		return y * (view - cos * norm) - norm * sqrtf(z);
	}


}

Flyscene::inters_point Flyscene::intersection(Eigen::Vector3f origin,
	Eigen::Vector3f dest) {

	std::vector<float> boxts;
	std::vector<Eigen::Vector3f> directionsb;
	std::vector<Eigen::Vector3f> normalsb;
	std::vector<Tucano::Face> facesb;
	for (int i = 0; i < bboxes.size(); i++) {
		std::vector<Tucano::Face> box = bboxes[i];
		std::vector<float> myCoords = makePlanes(box);
		std::vector<float> intersPoints;

		//cout << "AICI " << myCoords[0] << " " << myCoords[1] << " " << myCoords[2] << " " << myCoords[3] << " " << myCoords[4] << " " << myCoords[5] << endl;

		Eigen::Vector3f d = (dest - origin);
		float txmin = (myCoords[0] - origin.x()) / d.x();
		float tymin = (myCoords[1] - origin.y()) / d.y();
		float tzmin = (myCoords[2] - origin.z()) / d.z();
		float txmax = (myCoords[3] - origin.x()) / d.x();
		float tymax = (myCoords[4] - origin.y()) / d.y();
		float tzmax = (myCoords[5] - origin.z()) / d.z();

		//std::cout << "txmin: " << txmin << endl;
		//cout << "txmax: " << txmax << endl;
		float tinx = txmin < txmax ? txmin : txmax;
		float toutx = txmin > txmax ? txmin : txmax;
		float tiny = tymin < tymax ? tymin : tymax;
		float touty = tymin > tymax ? tymin : tymax;
		float tinz = tzmin < tzmax ? tzmin : tzmax;
		float toutz = tzmin > tzmax ? tzmin : tzmax;

		float tin = 0;
		float tout = 0;
		if (tinx >= tiny && tinx >= tinz)
			tin = tinx;
		else if (tiny >= tinx && tiny >= tinz)
			tin = tiny;
		else if (tinz >= tinx && tinz >= tiny)
			tin = tinz;

		if (toutx <= touty && toutx <= toutz)
			tout = toutx;
		else if (touty <= toutx && touty <= toutz)
			tout = touty;
		else if (toutz <= toutx && toutz <= touty)
			tout = toutz;
		/// if the ray hits our box
		if (!(tin > tout || tout < 0)) {

			//cout << "hit -> BOX: " << i << endl;

			Tucano::Shapes::Box repBox;
			Eigen::Vector3f boxorigin = Eigen::Vector3f((myCoords[0] + myCoords[3]) / 2, (myCoords[1] + myCoords[4]) / 2, (myCoords[2] + myCoords[5]) / 2);
			repBox = Tucano::Shapes::Box(myCoords[0] - myCoords[3], myCoords[1] - myCoords[4], myCoords[2] - myCoords[5]);

			repBox.modelMatrix()->translate(boxorigin);
			repBox.setColor({ static_cast <float> (rand()) / static_cast <float> (RAND_MAX), static_cast <float> (rand()) / static_cast <float> (RAND_MAX), static_cast <float> (rand()) / static_cast <float> (RAND_MAX), 1.0f });
			allBoxes.push_back(repBox);
			Eigen::Vector3f intersectionv;
			std::vector<float> ts;
			std::vector<Eigen::Vector3f> directions;
			std::vector<Eigen::Vector3f> normals;
			std::vector<Tucano::Face> faces;
			for (int i = 0; i < box.size(); ++i) {
				Tucano::Face face = box[i];

				/*std::cout << i << std::endl;*/
				float alpha;
				float beta;

				Eigen::Vector3f directionV = (dest - origin);
				Eigen::Vector3f facenormal = face.normal.normalized();
				//float distance = pow((pow(directionV.x(), 2) + pow(directionV.y(), 2) + pow(directionV.z(), 2)), 0.5);
				directionV.normalize();
				Eigen::Vector4f homogeneous = mesh.getShapeModelMatrix() * mesh.getVertex(face.vertex_ids[0]);
				Eigen::Matrix4f matrix = Eigen::Matrix4f(mesh.getShapeModelMatrix().matrix());
				//homogeneous = homogeneous.m * matrix;
				Eigen::Vector3f real = Eigen::Vector3f(homogeneous.x() / homogeneous.w(), homogeneous.y() / homogeneous.w(), homogeneous.z() / homogeneous.w());
				float distance = facenormal.dot(Eigen::Vector3f(homogeneous.x() / homogeneous.w(), homogeneous.y() / homogeneous.w(), homogeneous.z() / homogeneous.w()));
				float origin_normal = origin.dot(facenormal);
				float direction_normal = directionV.dot(facenormal);

				//check whether ray is parallel to plane
				if (fabs(direction_normal) < minCheck) {
					continue;
				}

				float t = (distance - origin_normal) / direction_normal;
				if (t <= 0) {
					continue;
				}

				intersectionv = origin + t * directionV;

				//check whether intersection is inside triangle
				std::vector<Eigen::Vector3f> vectors;
				for (int j = 0; j < 3; j++) {
					Eigen::Vector4f homogeneous = mesh.getShapeModelMatrix() * mesh.getVertex(face.vertex_ids[j]);
					Eigen::Vector3f real = Eigen::Vector3f(homogeneous.x() / homogeneous.w(), homogeneous.y() / homogeneous.w(), homogeneous.z() / homogeneous.w());
					vectors.push_back(real);
				}

				float v1x = vectors[0][0];
				float v1y = vectors[0][1];
				float v2x = vectors[1][0];
				float v2y = vectors[1][1];
				float v3x = vectors[2][0];
				float v3y = vectors[2][1];

				//std::cout << v1x << " " << v1y << " " << v2x << " " << v2y << " " << v3x << " " << v3y << std::endl;

				//alpha = (v1x * (v3y - v1y) + (intersectionv.y() - v1y) * (v3x - v1x) - (intersectionv.x() * (v3y - v1y)))
				//	/ ((v2y - v1y) * (v3x - v1x) - (v2x - v1x) * (v3y - v1y));
				//beta = (intersectionv.y() - v1y - alpha * (v2y - v1y))
				//	/ (v3y - v1y);

				//barycentric(intersectionv, vectors, alpha, beta);


				bool inTriangle = barycentric(intersectionv, vectors, alpha, beta);

				//std::cout << (alpha >= 0 && beta >= 0 && (alpha + beta) <= 1) << std::endl;
				//std::cout << inTriangle << std::endl;
				if (inTriangle) {
					/*std::cout << "Found intersection at" << std::endl << intersectionv << std::endl;
					std::cout << "distance: " << distance << " t: " << t << " alpha: " << alpha << " beta: " << beta << std::endl;*/
					ts.push_back(t);
					normals.push_back(facenormal);
					directions.push_back(directionV);
					faces.push_back(face);
				}
			}
			if (ts.size() == 0) {
				//return Flyscene::inters_point{ false, Eigen::Vector3f(), Tucano::Face() };
				continue;
			}
			else {
				//calc correct intersection point(closest to the camera)
				float min = *std::min_element(ts.begin(), ts.end());
				std::vector<float>::iterator indexit = std::find(ts.begin(), ts.end(), min);
				float index = indexit - ts.begin();
				//get material of corresponding face
				Eigen::Vector3f direction = directions[index];
				//get normalv
				/*std::cout << "normal before struct: " << normals[index] << std::endl;
				std::cout << "intersection point before struct: " << origin + min * direction << std::endl;*/
				Eigen::Vector3f point = origin + min * direction;
				Tucano::Face face = faces[index];
				//return Flyscene::inters_point{ true, point, face };
				boxts.push_back(ts[index]);
				directionsb.push_back(direction);
				facesb.push_back(face);
			}

		}
		else {
			continue;
		}
	}

	if (boxts.size() == 0) {
		return Flyscene::inters_point{ false, Eigen::Vector3f(), Tucano::Face() };
	}

	float min = *std::min_element(boxts.begin(), boxts.end());
	std::vector<float>::iterator indexit = std::find(boxts.begin(), boxts.end(), min);
	float index = indexit - boxts.begin();
	//get material of corresponding face
	Eigen::Vector3f direction = directionsb[index];
	/*std::cout << "normal before struct: " << normals[index] << std::endl;
	std::cout << "intersection point before struct: " << origin + min * direction << std::endl;*/
	Eigen::Vector3f point = origin + min * direction;
	Tucano::Face face = facesb[index];
	return Flyscene::inters_point{ true, point, face };
}

float Flyscene::clamp(float x, float low, float high) {
	return x < low ? low : x > high ? high : x;
}

bool Flyscene::barycentric(Eigen::Vector3f p, std::vector<Eigen::Vector3f> vectors, float& alpha, float& beta) {
	Eigen::Vector3f v0 = vectors[1] - vectors[0];
	Eigen::Vector3f v1 = vectors[2] - vectors[0];
	Eigen::Vector3f v2 = p - vectors[0];

	float d00 = v0.dot(v0);
	float d01 = v0.dot(v1);
	float d11 = v1.dot(v1);
	float d20 = v2.dot(v0);
	float d21 = v2.dot(v1);
	float denom = d00 * d11 - d01 * d01;

	alpha = (d11 * d20 - d01 * d21) / denom;
	if (alpha < 0) return false;

	beta = (d00 * d21 - d01 * d20) / denom;
	return (beta < 0 || alpha + beta > 1) ? false : true;
}

Eigen::Vector3f Flyscene::reflect(Eigen::Vector3f incoming, Eigen::Vector3f normal)
{
	/*std::cout << "incoming" << incoming << std::endl;
	std::cout << "normal" << normal << std::endl;*/
	normal.normalize();
	incoming.normalize();
	Eigen::Vector3f reflection = incoming - 2 * (incoming.dot(normal) * normal);
	return reflection.normalized();
}

Eigen::Vector3f Flyscene::shade(int level, int maxlevel, Eigen::Vector3f intersection, Eigen::Vector3f ray, Tucano::Face face, float shadowratio) {
	if (intersection == Eigen::Vector3f()) {
		return Eigen::Vector3f(backgroundColor.x(), backgroundColor.y(), backgroundColor.z());
	}
	if (level < maxlevel) {
		return shadowratio * directColor(intersection, ray, face) + reflectColor(level, intersection, ray, face);
	}
	return shadowratio * directColor(intersection, ray, face);
}

Eigen::Vector3f Flyscene::directColor(Eigen::Vector3f p, Eigen::Vector3f ray, Tucano::Face face) {
	//ambient term
	Eigen::Vector3f ka;
	Eigen::Vector3f kd;
	Eigen::Vector3f ks;
	float shininess;

	if (face.material_id != -1) {
		ka = materials[face.material_id].getAmbient();
		kd = materials[face.material_id].getDiffuse();
		ks = materials[face.material_id].getSpecular();
		shininess = materials[face.material_id].getShininess();
	}
	else {
		//fallback values to avoid errors for objs without materials
		ka = Eigen::Vector3f(0, 0, 0);
		kd = Eigen::Vector3f(1, 0.5, 0);
		ks = Eigen::Vector3f(0.5, 0.5, 0.5);
		shininess = 10;
	}
	Eigen::Vector3f lightIntensity = Eigen::Vector3f(1.0, 1.0, 1.0);
	Eigen::Vector3f ambient = lightIntensity.cwiseProduct(ka);
	Eigen::Vector3f normal = face.normal.normalized();
	Eigen::Vector3f diffuse = Eigen::Vector3f(0, 0, 0);
	Eigen::Vector3f specular = Eigen::Vector3f(0, 0, 0);
	//calc for multiple lights 
	for (int k = 0; k < lights.size(); k++) {
		//diffuse
		Eigen::Vector3f lightDir = (lights[k] - p).normalized();
		float diffuseDot = lightDir.dot(normal);
		float diffuseBounded = (diffuseDot > 0.0) ? diffuseDot : 0.0;
		diffuse += lightIntensity.cwiseProduct(kd) * diffuseBounded;

		//specular term
		Eigen::Vector3f reflectV = reflect(lightDir, face.normal.normalized()).normalized();
		Eigen::Vector3f cameraV = flycamera.getCenter() - p;
		float specularDot = (reflectV.normalized()).dot(cameraV.normalized());

		float specularBounded = (specularDot > 0.0) ? specularDot : 0.0;
		specular += lightIntensity.cwiseProduct(ks) * std::pow(specularBounded, shininess);
	}
	return ambient + diffuse + specular;
}

Eigen::Vector3f Flyscene::reflectColor(int level, Eigen::Vector3f intersectionP, Eigen::Vector3f ray, Tucano::Face face) {
	Tucano::Material::Mtl current_material = materials[face.material_id];
	Eigen::Vector3f specular = current_material.getSpecular();

	Eigen::Vector3f reflectV = reflect(ray, face.normal);
	//add bias to avoid reflection on the same object
	inters_point newIntersection = intersection(intersectionP + reflectV * REFLECT_BIAS, reflectV + intersectionP);
	//if reflection doesnt intersect then return background color
	if (newIntersection.intersected == false && !specular.isZero()) {
		return specular.cwiseProduct(Eigen::Vector3f(backgroundColor.x(), backgroundColor.y(), backgroundColor.z()));
	}
	//check if material is reflective, if so then go calculate recursion
	if (!specular.isZero()) {
		float shadowratio = shadowRatio(newIntersection.point, newIntersection.face);
		return specular.cwiseProduct(Flyscene::shade(++level, MAX_REFLECT, newIntersection.point, newIntersection.point - intersectionP, newIntersection.face, shadowratio));
	}
	return Eigen::Vector3f(0, 0, 0);
}

Eigen::Vector3f Flyscene::refractColor(int level, Eigen::Vector3f intersectionP, Eigen::Vector3f ray, Tucano::Face face) {
	Tucano::Material::Mtl current_material = materials[face.material_id];
	Eigen::Vector3f specular = current_material.getSpecular();

	float index = current_material.getOpticalDensity();
	float transparency = current_material.getDissolveFactor();

	Eigen::Vector3f refractV = refractionV(ray, face.normal, index);
	inters_point newIntersection = intersection(intersectionP + refractV * REFLECT_BIAS, refractV + intersectionP);

	if (newIntersection.intersected == false && transparency != 1.0)
	{
		return  (1 - transparency) * (Eigen::Vector3f(1, 1, 1) - specular).cwiseProduct(Eigen::Vector3f(backgroundColor.x(), backgroundColor.y(), backgroundColor.z()));
	}

	if (transparency != 1.0)
	{
		float shadowratio = shadowRatio(newIntersection.point, newIntersection.face);
		return(1 - transparency) * (Eigen::Vector3f(1, 1, 1) - specular).cwiseProduct(Flyscene::shade(++level, MAX_REFLECT, newIntersection.point, newIntersection.point - intersectionP, newIntersection.face, shadowratio));
	}
	return Eigen::Vector3f(0, 0, 0);
}

std::vector<float> Flyscene::makePlanes(std::vector<Tucano::Face> box) {
	std::vector<Eigen::Vector4f> vertices;

	for (int x = 0; x < box.size(); x++) {
		Tucano::Face face = box[x];
		for (int z = 0; z < 3; z++) {

			//vertices.push_back(mesh.getVertex(face.vertex_ids[z]));

			Eigen::Vector4f homogeneous = shapeModelMatrix * mesh.getVertex(face.vertex_ids[z]);
			Eigen::Vector4f real = Eigen::Vector4f(homogeneous.x() / homogeneous.w(), homogeneous.y() / homogeneous.w(), homogeneous.z() / homogeneous.w(), 1.0);
			vertices.push_back(real);
		}

	}

	Eigen::Vector4f firstVertex = vertices[0];
	float xmax = firstVertex.x();
	float xmin = firstVertex.x();
	float ymax = firstVertex.y();
	float ymin = firstVertex.y();
	float zmax = firstVertex.z();
	float zmin = firstVertex.z();
	for (int x = 0; x < vertices.size(); x++) {
		Eigen::Vector4f currentVertex = vertices[x];
		if (xmin > currentVertex.x()) xmin = currentVertex.x();
		if (xmax < currentVertex.x()) xmax = currentVertex.x();

		if (ymin > currentVertex.y()) ymin = currentVertex.y();
		if (ymax < currentVertex.y()) ymax = currentVertex.y();

		if (zmin > currentVertex.z()) zmin = currentVertex.z();
		if (zmax < currentVertex.z()) zmax = currentVertex.z();
	}

	std::vector<float> newVector;
	newVector.push_back(xmin);
	newVector.push_back(ymin);
	newVector.push_back(zmin);
	newVector.push_back(xmax);
	newVector.push_back(ymax);
	newVector.push_back(zmax);

	return newVector;
}

std::vector<std::vector<Tucano::Face>> Flyscene::subdivide() {
	std::vector<Tucano::Face> primarybb;
	int totalFaces = mesh.getNumberOfFaces();
	Eigen::Vector3f avg = Eigen::Vector3f(0.0, 0.0, 0.0);
	Eigen::Vector3f sum = Eigen::Vector3f(0.0, 0.0, 0.0);
	Tucano::Face face;

	for (int i = 0; i < totalFaces; ++i) {
		face = mesh.getFace(i);
		primarybb.push_back(face);
		for (int j = 0; j < 3; ++j) {
			int id = face.vertex_ids[j];

			Eigen::Vector4f homogeneous = shapeModelMatrix * mesh.getVertex(face.vertex_ids[j]);
			Eigen::Vector3f real = Eigen::Vector3f(homogeneous.x() / homogeneous.w(), homogeneous.y() / homogeneous.w(), homogeneous.z() / homogeneous.w());

			sum += real;

		}
		sum /= 3;
		avg += sum;
		sum = Eigen::Vector3f(0.0, 0.0, 0.0);
	}
	avg /= mesh.getNumberOfFaces();
	std::vector<float> bounds = makePlanes(primarybb);
	int boxSize = totalFaces;
	return split(bounds, primarybb, avg);
}

std::vector<std::vector<Tucano::Face>> Flyscene::split(std::vector<float> bounds, std::vector<Tucano::Face> bb, Eigen::Vector3f avg) {
	std::vector<Tucano::Face> bb1;
	std::vector<Tucano::Face> bb2;

	if (bb.size() <= AMOUNT_FACES) {
		std::vector<std::vector<Tucano::Face>> result;
		result.push_back(bb);
		return result;
	}

	float xdiff = bounds[0] - bounds[3];
	xdiff = xdiff < 0 ? -xdiff : xdiff;
	float ydiff = bounds[1] - bounds[4];
	ydiff = ydiff < 0 ? -ydiff : ydiff;
	float zdiff = bounds[2] - bounds[5];
	zdiff = zdiff < 0 ? -zdiff : zdiff;

	int cnt1 = 0;
	int cnt2 = 0;
	Eigen::Vector3f avg1 = Eigen::Vector3f(0.0, 0.0, 0.0);
	Eigen::Vector3f avg2 = Eigen::Vector3f(0.0, 0.0, 0.0);
	Eigen::Vector3f temp = Eigen::Vector3f(0.0, 0.0, 0.0);

	if (xdiff >= ydiff && xdiff >= zdiff) {
		for (int i = 0; i < bb.size(); ++i) {
			//int vid0 = bb.at(i).vertex_ids[0];
			//int vid1 = bb.at(i).vertex_ids[1];
			//int vid2 = bb.at(i).vertex_ids[2];
			//temp = mesh.getVertex(vid0) + mesh.getVertex(vid1) + mesh.getVertex(vid2);
			temp = Eigen::Vector3f(0.0, 0.0, 0.0);
			Tucano::Face face = bb[i];
			for (int j = 0; j < 3; ++j) {
				int id = face.vertex_ids[j];

				Eigen::Vector4f homogeneous = shapeModelMatrix * mesh.getVertex(face.vertex_ids[j]);

				Eigen::Vector3f real = Eigen::Vector3f(homogeneous.x() / homogeneous.w(), homogeneous.y() / homogeneous.w(), homogeneous.z() / homogeneous.w());


				temp += real;
			}

			temp /= 3;
			if (temp.x() + 0.000001 < avg.x()) {
				avg1 += temp;
				cnt1++;
				bb1.push_back(bb.at(i));
			}
			else if (temp.x() - 0.000001 > avg.x()){

				avg2 += temp;
				cnt2++;
				bb2.push_back(bb.at(i));
			}
			else {
				if (bb1.size() > bb2.size()) {
					avg2 += temp;
					cnt2++;
					bb2.push_back(bb.at(i));
				}
				else {
					avg1 += temp;
					cnt1++;
					bb1.push_back(bb.at(i));
				}
			}
			//temp = Eigen::Vector4f(0, 0, 0, 0);
		}
	}
	else if (ydiff >= xdiff && ydiff >= zdiff) {
		for (int i = 0; i < bb.size(); ++i) {
			//int vid0 = bb.at(i).vertex_ids[0];
			//int vid1 = bb.at(i).vertex_ids[1];
			//int vid2 = bb.at(i).vertex_ids[2];
			//temp = mesh.getVertex(vid0) + mesh.getVertex(vid1) + mesh.getVertex(vid2);
			temp = Eigen::Vector3f(0.0, 0.0, 0.0);
			Tucano::Face face = bb[i];
			for (int j = 0; j < 3; ++j) {
				int id = face.vertex_ids[j];

				Eigen::Vector4f homogeneous = shapeModelMatrix * mesh.getVertex(face.vertex_ids[j]);
				Eigen::Vector3f real = Eigen::Vector3f(homogeneous.x() / homogeneous.w(), homogeneous.y() / homogeneous.w(), homogeneous.z() / homogeneous.w());


				temp += real;
			}
			temp /= 3;
			if (temp.y() + 0.000001 < avg.y()) {
				avg1 += temp;
				cnt1++;
				bb1.push_back(bb.at(i));
			}
			else if (temp.y() - 0.000001 > avg.y()){
				avg2 += temp;
				cnt2++;
				bb2.push_back(bb.at(i));
			}
			else {
				if (bb1.size() > bb2.size()) {
					avg2 += temp;
					cnt2++;
					bb2.push_back(bb.at(i));
				}
				else {
					avg1 += temp;
					cnt1++;
					bb1.push_back(bb.at(i));
				}
			}
			//temp = Eigen::Vector4f(0, 0, 0, 0);
		}
	}
	else {
		for (int i = 0; i < bb.size(); ++i) {
			//int vid0 = bb.at(i).vertex_ids[0];
			//int vid1 = bb.at(i).vertex_ids[1];
			//int vid2 = bb.at(i).vertex_ids[2];
			//temp = mesh.getVertex(vid0) + mesh.getVertex(vid1) + mesh.getVertex(vid2);
			temp = Eigen::Vector3f(0.0, 0.0, 0.0);
			Tucano::Face face = bb[i];
			for (int j = 0; j < 3; ++j) {
				int id = face.vertex_ids[j];
				Eigen::Vector4f homogeneous = shapeModelMatrix * mesh.getVertex(face.vertex_ids[j]);
				Eigen::Vector3f real = Eigen::Vector3f(homogeneous.x() / homogeneous.w(), homogeneous.y() / homogeneous.w(), homogeneous.z() / homogeneous.w());


				temp += real;
			}
			temp /= 3;
			if (temp.z() + 0.000001 < avg.z()) {
				avg1 += temp;
				cnt1++;
				bb1.push_back(bb.at(i));
			}
			else if (temp.z() - 0.000001 > avg.z()){

				avg2 += temp;
				cnt2++;
				bb2.push_back(bb.at(i));
			}
			else {
				if (bb1.size() > bb2.size()) {
					avg2 += temp;
					cnt2++;
					bb2.push_back(bb.at(i));
				}
				else {
					avg1 += temp;
					cnt1++;
					bb1.push_back(bb.at(i));
				}
			}
			//temp = Eigen::Vector4f(0, 0, 0, 0);
		}
	}
	if (cnt1)
		avg1 /= cnt1;
	if (cnt2)
		avg2 /= cnt2;
	std::vector<std::vector<Tucano::Face>> result1;
	std::vector<std::vector<Tucano::Face>> result2;
	cout << "bb1 " << bb1.size() << endl;
	cout << "bb2 " << bb2.size() << endl;
	if (bb1.size() > AMOUNT_FACES) {
		std::vector<float> bounds1 = makePlanes(bb1);
		result1 = split(bounds1, bb1, avg1);
	}
	else if (bb1.size() > 0) {
		result1.push_back(bb1);
	}

	if (bb2.size() > AMOUNT_FACES) {
		std::vector<float> bounds2 = makePlanes(bb2);
		result2 = split(bounds2, bb2, avg2);
	}
	else if (bb2.size() > 0) {
		result2.push_back(bb2);
	}

	result1.insert(result1.end(), result2.begin(), result2.end());
	return result1;
}


float Flyscene::shadowRatio(Eigen::Vector3f intersectionP, Tucano::Face face) {

	//Counter for how many rays reach the light
	float raysReachLight = 0;

	for (int i = 0; i < lights.size(); i++) {
		//Each iterations shoots a ray to light
		if (SOFT_SHADOWS) {
			for (int j = 0; j < SHADOW_SMOOTHNESS; j++)
			{
				//Create two random floats between -0.075 and 0.075
				//The range will be hardcoded unless we find a way to get the radius of the light
				double randX = (rand() % 16) / 100 - 0.075;
				double randY = (rand() % 16) / 100 - 0.075;
				double randZ = (rand() % 16) / 100 - 0.075;

				Eigen::Vector3f offset = Eigen::Vector3f(randX, randY, randZ);

				//Shoot a ray from hit point to light center shifted by the offset vector
				inters_point rayToLight = intersection(intersectionP + face.normal * shadowBias, (offset + lights[i]));

				//See if ray reaches light. Increment counter if it does
				if (!rayToLight.intersected) {
					raysReachLight++;
				}
			}
		}
		else {//Use hard shadows
			inters_point rayToLight = intersection(intersectionP + face.normal * shadowBias, lights[i]);
			if (!rayToLight.intersected)
			{
				raysReachLight++;
			}
		}
	}
	if (SOFT_SHADOWS)
	{
		return raysReachLight / (float)(SHADOW_SMOOTHNESS * lights.size());

	}
	else
	{
		return  (float)(raysReachLight / (float)lights.size());
	}
}


