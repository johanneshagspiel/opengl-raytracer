#include "flyscene.hpp"
#include <GLFW/glfw3.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <future>

static const int threshhold = 345;

//Performance Results
static int boxesHit = 0;
static int raysSend = 0;
static int interceptions = 0;

// User input
static int levelReflection = 1;

void Flyscene::initialize(int width, int height) {
    // initiliaze the Phong Shading effect for the Opengl Previewer
    phong.initialize();


    // set the camera's projection matrix
    flycamera.setPerspectiveMatrix(60.0, width / (float) height, 0.1f, 100.0f);
    flycamera.setViewport(Eigen::Vector2f((float) width, (float) height));

    // load the OBJ file and materials
//    Tucano::MeshImporter::loadObjFile(mesh, materials, "resources/models/shade-cube.obj");
//    Tucano::MeshImporter::loadObjFile(mesh, materials, "resources/models/shade-stick.obj");
//    Tucano::MeshImporter::loadObjFile(mesh, materials, "resources/models/mirror-stick.obj");
    Tucano::MeshImporter::loadObjFile(mesh, materials, "resources/models/test.obj");
//    Tucano::MeshImporter::loadObjFile(mesh, materials, "resources/models/test2.obj");
//    Tucano::MeshImporter::loadObjFile(mesh, materials, "resources/models/scene.obj");

    // normalize the model (scale to unit cube and center at origin)
    mesh.normalizeModelMatrix();


    // pass all the materials to the Phong Shader
    for (int i = 0; i < materials.size(); ++i)
        phong.addMaterial(materials[i]);


    // set the color and size of the sphere to represent the light sources
    // same sphere is used for all sources
    lightrep.setColor(Eigen::Vector4f(1.0, 1.0, 0.0, 1.0));
    lightrep.setSize(0.15);

    // create a first ray-tracing light source at some random position
    lights.push_back(Eigen::Vector4f(-0.5, 0.85, 1, 0.25));

    //set the color and the size of our debugOrbRep
    debugOrbRep.setColor(Eigen::Vector4f(1.0, 0.0, 0.0, 1.0));
    debugOrbRep.setSize(.08);

    // scale the camera representation (frustum) for the ray debug
    camerarep.shapeMatrix()->scale(0.2);


    // the debug ray is a cylinder, set the radius and length of the cylinder
    ray.setSize(0.005, 10.0);
    

    boxes = firstBox(mesh);

    for (int i = 0; i < boxes.size(); i++) {
        boxbounds.push_back(getBoxLimits(boxes[i], mesh));
    }

    glEnable(GL_DEPTH_TEST);

    camSpeed = 1;

}

void Flyscene::paintGL(void) {

    // update the camera view matrix with the last mouse interactions
    flycamera.updateViewMatrix();
    Eigen::Vector4f viewport = flycamera.getViewport();

    // clear the screen and set background color
    glClearColor(0.9, 0.9, 0.9, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // position the scene light at the last ray-tracing light source
    scene_light.resetViewMatrix();
    scene_light.viewMatrix()->translate(-lights.back().head<3>());

    // render the scene using OpenGL and one light source
    phong.render(mesh, flycamera, scene_light);


    // render the ray and camera representation for ray debug
    ray.render(flycamera, scene_light);
	
	for (int i = 0; i < reflection_rays.size(); i++) {
		reflection_rays[i].render(flycamera, scene_light);
		refraction_rays[0].render(flycamera, scene_light);
	}

    camerarep.render(flycamera, scene_light);

    // render ray tracing light sources as yellow spheres
    for (int i = 0; i < lights.size(); ++i) {
        lightrep.resetModelMatrix();
        lightrep.modelMatrix()->translate(lights[i].head<3>());
        lightrep.render(flycamera, scene_light);
    }

    //render the debugOrbRep's as red spheres
    for (int i = 0; i < debugOrbs.size(); i++) {
        debugOrbRep.resetModelMatrix();
        debugOrbRep.modelMatrix()->translate(debugOrbs[i]);
        debugOrbRep.render(flycamera, scene_light);
    }

    // render coordinate system at lower right corner
    flycamera.renderAtCorner();
}

void Flyscene::simulate(GLFWwindow *window) {
    // Update the camera.
    // NOTE(mickvangelderen): GLFW 3.2 has a problem on ubuntu where some key
    // events are repeated: https://github.com/glfw/glfw/issues/747. Sucks.
    float dx = (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS ? 1.0 : 0.0) -
               (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS ? 1.0 : 0.0);

    float dy = (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS ? 1.0 : 0.0) -
               (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS ? 1.0 : 0.0);

    float dz = (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS ? 1.0 : 0.0) -
               (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS ? 1.0 : 0.0);

    float ds = (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS ? 1.0 : 0.0) -
               (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS ? 1.0 : 0.0);

    camSpeed *= (1 + ds * 0.1);

    flycamera.translate(camSpeed * dx, camSpeed * dy, camSpeed * dz);
}


void Flyscene::raytraceScene(int width, int height) {

	std::cout << "How many levels of reflection do you want ?" << std::endl;
	std::cin >> levelReflection;

    std::cout << "ray tracing ..." << std::endl;

    auto start = chrono::steady_clock::now();

    std::cout.setstate(std::ios_base::failbit);

    // if no width or height passed, use dimensions of current viewport
    Eigen::Vector2i image_size(width, height);
    if (width == 0 || height == 0) {
        image_size = flycamera.getViewportSize();
    }

    Eigen::Vector3f center = Eigen::Vector3f(0, 0, 0); //mesh.getCentroid(); broken
    float sphere_r = 1.0f;//mesh.getBoundingSphereRadius();broken

    // create 2d vector to hold pixel colors and resize to match image size
    vector<vector<Eigen::Vector3f>> pixel_data;
    pixel_data.resize(image_size[1]);
    for (int i = 0; i < image_size[1]; ++i)
        pixel_data[i].resize(image_size[0]);

    // origin of the ray is always the camera center
    Eigen::Vector3f origin = flycamera.getCenter();
    Eigen::Vector3f screen_coords;

    // for every pixel shoot a ray from the origin through the pixel coords
    float progress = 0;
    float remaining = 0;
    for (int j = 0; j < image_size[1]; ++j) {
        for (int i = 0; i < image_size[0]; ++i) {
            if(i%10==0){
                progress = ((1.0 * j / image_size[1]) + (1.0 * i / image_size[0] / image_size[1]));
                float time = chrono::duration<double, milli>(chrono::steady_clock::now() - start).count() / 1000;
                remaining = time/progress - time;
            }
            printf("\r[x:%-3i y:%-3i] %5.1f%% done %6.1fs remaining\t\t", j, i, 100*progress, remaining);

            screen_coords = flycamera.screenToWorld(Eigen::Vector2f(i, j));
            // write result to pixel data
            pixel_data[j][i] = traceRay(origin, screen_coords, boxes, boxbounds, center, sphere_r);
        }
    }
    printf("\n");

    // write the ray tracing result to a PPM image
	raysSend = image_size[0] * image_size[1];
    Tucano::ImageImporter::writePPMImage("result.ppm", pixel_data);

    std::cout.clear();
    auto end = chrono::steady_clock::now();
    auto diff = end - start;

    std::cout << chrono::duration<double, milli>(diff).count() / 1000 << " seconds" << endl;

    std::cout << "ray tracing done! " << std::endl;
	std::cout << "Boxes hit: "<< boxesHit << std::endl;
	std::cout << "Rays send: " << raysSend << std::endl;
	std::cout << "Interceptions: " << interceptions << std::endl;
}

float distance(Eigen::Vector3f a, Eigen::Vector3f b) {
	return sqrt((a - b).squaredNorm());
}


//Intersection with plane
// src = slights, https://stackoverflow.com/questions/23975555/how-to-do-ray-plane-intersection
auto intersectPlane(Eigen::Vector3f start, Eigen::Vector3f to, Eigen::Vector3f normal, Eigen::Vector3f p) {

    float denom = normal.dot(to);
    Eigen::Vector3f point;

    struct result {
        bool inter;
        float t;
        Eigen::Vector3f point;
    };

    if (abs(denom) > 0) {

        float t = (p - start).dot(normal) / denom;
		Eigen::Vector3f xyz = (start + t * to);

        if (t < 0.000001f) {
            return result{false, 0, xyz};
        }

        //if intersection point and start is same doesnt count for shade
        if (xyz.x() == start.x() && xyz.y() == start.y() && xyz.z() == start.z()) {
            return result{false, 0, p};
        }

        return result{true, t, xyz};
    }

    return result{false, 0, p};
}


//Intersection with triangle
//give normalized vectors
// src = https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
auto intersectTriange(Eigen::Vector3f start, Eigen::Vector3f to, Tucano::Face face, Tucano::Mesh mesh) {

    const Eigen::Vector4f vec40 = (mesh.getShapeModelMatrix() * mesh.getVertex(face.vertex_ids[0]));
    const Eigen::Vector4f vec41 = (mesh.getShapeModelMatrix() * mesh.getVertex(face.vertex_ids[1]));
    const Eigen::Vector4f vec42 = (mesh.getShapeModelMatrix() * mesh.getVertex(face.vertex_ids[2]));

    Eigen::Vector3f facenormal = face.normal;

    Eigen::Vector3f v0 = (vec40 / vec40.w()).head<3>();
    Eigen::Vector3f v1 = (vec41 / vec41.w()).head<3>();
    Eigen::Vector3f v2 = (vec42 / vec42.w()).head<3>();

    struct result {
        bool inter;
        Tucano::Face face;
        Tucano::Mesh mesh;
        Eigen::Vector3f hit;
    };

    //std::cout << "Tiangle Intersect";

    auto planeinter = intersectPlane(start, to, facenormal, v0);

    if (planeinter.inter) {
        Eigen::Vector3f p_onPlane = planeinter.point;

        Eigen::Vector3f per;

        Eigen::Vector3f edge0 = v1 - v0;
        Eigen::Vector3f vp0 = p_onPlane - v0;
        per = edge0.cross(vp0);
        if (facenormal.dot(per) < 0.000001f) {
            return result{false, face, mesh, p_onPlane};
        }


        Eigen::Vector3f edge1 = v2 - v1;
        Eigen::Vector3f vp1 = p_onPlane - v1;
        per = edge1.cross(vp1);
        if (facenormal.dot(per) < 0.000001f) {
            return result{false, face, mesh, p_onPlane};
        }


        Eigen::Vector3f edge2 = v0 - v2;
        Eigen::Vector3f vp2 = p_onPlane - v2;
        per = edge2.cross(vp2);
        if (facenormal.dot(per) < 0.000001f) {
            return result{false, face, mesh, p_onPlane};
        }

        //std::cout << "TRIANGEL INTERSECTION FOUND " << std::endl;
        return result{true, face, mesh, p_onPlane};

    }

    return result{false, face, mesh, v0};

}

//Intersection with sphere
// src = https://www.scratchapixel.com/code.php?id=8&origin=/lessons/3d-basic-rendering/ray-tracing-overview
auto Flyscene::intersectSphere(Eigen::Vector3f origin, Eigen::Vector3f direction, Eigen::Vector3f center, float radius) {

    Eigen::Vector3f light = origin - center;
    float a = direction.dot(direction);
    float b = 2 * direction.dot(light);
    float c = light.dot(light) - (radius * radius);
	float solution1;
	float solution2;

    return (Flyscene::solveQuadraticEquation(a, b, c, solution1, solution2));
}

//Helper function to solving a quadratic equation
// src = https://www.scratchapixel.com/code.php?id=8&origin=/lessons/3d-basic-rendering/ray-tracing-overview
// src = https://www.programiz.com/cpp-programming/examples/quadratic-roots
bool Flyscene::solveQuadraticEquation(float &a, float &b, float &c, float & solution1, float & solution2) {
    float discriminant = b * b - 4 * a * c;

	if (discriminant > 0)
	{
		solution1 = ((-b + sqrt(discriminant)) / (2 * a));
		solution2 = ((b + sqrt(discriminant)) / (2 * a));
		return true;
	}
	else if (discriminant == 0) {
		solution1 = ((-b + sqrt(discriminant)) / (2 * a));
		solution2 = ((-b + sqrt(discriminant)) / (2 * a));
		return true;
	}
	else {
		return false;
	}
}


//Object for git info
class HitData {
    // Access specifier
public:
    // Data Members
    Tucano::Face face;
    Eigen::Vector3f hit;

    // Member Functions()
    Eigen::Vector3f gethit() {
        return hit;
    }
};

//gives the min and max of the box AND THE MEAN
std::vector<Eigen::Vector3f> Flyscene::getBoxLimits(std::vector<Tucano::Face> box, Tucano::Mesh mesh) {

    std::vector<Eigen::Vector3f> vecs;
    Eigen::Vector3f vec;
    Eigen::Vector4f vec4;

    if (box.size() == 0) {
        return vecs;

    }

    std::vector<GLuint> vecsofface;
    //std::cout << "sasa";

    for (int i = 0; i < box.size(); i++) {
        for (int a = 0; a < box[i].vertex_ids.size(); a++) {
            vecsofface.push_back(box[i].vertex_ids[a]);
        }
    }
    //std::cout << "sasa";

    float mean_x = 0;
    float mean_y = 0;
    float mean_z = 0;

    vec4 = mesh.getShapeModelMatrix() * mesh.getVertex(vecsofface[0]);
    vec = (vec4 / vec4.w()).head<3>();

    float min_x = vec.x();
    float max_x = vec.x();

    float min_y = vec.y();
    float max_y = vec.y();

    float min_z = vec.z();
    float max_z = vec.z();

    for (int i = 0; i < vecsofface.size(); i++) {
        Eigen::Vector4f v4 = mesh.getShapeModelMatrix() * mesh.getVertex(vecsofface[i]);
        Eigen::Vector3f v = (v4 / v4.w()).head<3>();;

        mean_x = mean_x + v.x();
        mean_y = mean_y + v.y();
        mean_z = mean_z + v.z();

        if (min_x > v.x()) {
            min_x = v.x();
        }
        if (max_x < v.x()) {
            max_x = v.x();
        }
        if (min_y > v.y()) {
            min_y = v.y();
        }
        if (max_y < v.y()) {
            max_y = v.y();
        }
        if (min_z > v.z()) {
            min_z = v.z();
        }
        if (max_z < v.z()) {
            max_z = v.z();
        }
    }
    mean_x = mean_x / vecsofface.size();
    mean_y = mean_y / vecsofface.size();
    mean_z = mean_z / vecsofface.size();

    vecs.push_back(Eigen::Vector3f(min_x, min_y, min_z));
    vecs.push_back(Eigen::Vector3f(max_x, max_y, max_z));
    vecs.push_back(Eigen::Vector3f(mean_x, mean_y, mean_z));
    return vecs;
}


//recursuve box
void
Flyscene::createboxes(std::vector<Tucano::Face> box, Tucano::Mesh mesh, std::vector<std::vector<Tucano::Face>> &boxes) {
    // box is a vector with faces
    // boxes is a vector with boxes so vector -> vector(faces)
    std::vector<Tucano::Face> box1;
    std::vector<Tucano::Face> box2;

    //if box has less that 345 faces then push it to boxes
    // because if lower gets stucked
    if (box.size() < threshhold) {
        boxes.push_back(box);
        return;
    }

    //std::cout << "a" << std::endl;
    // box lim is the min max of box
    std::vector<Eigen::Vector3f> boxlim = Flyscene::getBoxLimits(box, mesh);

    float xdiff = boxlim[1].x() - boxlim[0].x();
    float ydiff = boxlim[1].y() - boxlim[0].y();
    float zdiff = boxlim[1].z() - boxlim[0].z();

    float cut;

    //std::cout << "red1 " << std::endl;
    //if x diff is the highest cut from there
    if (xdiff > ydiff && xdiff > zdiff) {
        //contains the mean
        cut = boxlim[2].x();

        //std::cout << " cut from x " << std::endl;
        for (int i = 0; i < box.size(); i++) {
            bool box1con = false;
            bool box2con = false;
            //std::cout << box.size() << std::endl;
            for (int a = 0; a < box[i].vertex_ids.size(); a++) {
                Eigen::Vector4f v4 = mesh.getShapeModelMatrix() * mesh.getVertex(box[i].vertex_ids[a]);
                Eigen::Vector3f v = (v4 / v4.w()).head<3>();;
                if (v.x() < cut) {
                    if (!box1con) {
                        box1.push_back(box[i]);
                        box1con = true;
                    }
                } else {
                    //changed this the same as the code above
                    if (!box2con) {
                        box2.push_back(box[i]);
                        box2con = true;
                    }
                }
            }
        }
        //if y dif is highest cut from y
    } else if (ydiff > xdiff && ydiff > zdiff) {
        //std::cout << " cut from y " << std::endl;
        cut = boxlim[2].y();
        //std::cout << box.size() << std::endl;
        for (int i = 0; i < box.size(); i++) {
            bool box1con = false;
            bool box2con = false;

            for (int a = 0; a < box[i].vertex_ids.size(); a++) {
                Eigen::Vector4f v4 = mesh.getShapeModelMatrix() * mesh.getVertex(box[i].vertex_ids[a]);
                Eigen::Vector3f v = (v4 / v4.w()).head<3>();;
                if (v.y() < cut) {
                    if (!box1con) {
                        box1.push_back(box[i]);
                        box1con = true;
                    }
                } else {
                    //changed this the same as the code above
                    if (!box2con) {
                        box2.push_back(box[i]);
                        box2con = true;
                    }
                }
            }
        }
        //if z dif is highest cut from z
    } else {
        //std::cout << " cut from z " << std::endl;
        cut = boxlim[2].z();
        //std::cout << box.size() << std::endl;
        for (int i = 0; i < box.size(); i++) {
            bool box1con = false;
            bool box2con = false;
            for (int a = 0; a < box[i].vertex_ids.size(); a++) {
                Eigen::Vector4f v4 = mesh.getShapeModelMatrix() * mesh.getVertex(box[i].vertex_ids[a]);
                Eigen::Vector3f v = (v4 / v4.w()).head<3>();;
                if (v.z() < cut) {
                    if (!box1con) {
                        box1.push_back(box[i]);
                        box1con = true;
                    }
                } else {
                    //changed this the same as the code above
                    if (!box2con) {
                        box2.push_back(box[i]);
                        box2con = true;
                    }
                }
            }
        }
    }
    //std::cout << "BOX1" << box1.size() << std::endl;
    createboxes(box1, mesh, boxes);
    //std::cout << "BOX2" << box2.size() << std::endl;
    createboxes(box2, mesh, boxes);

}

//first box
std::vector<std::vector<Tucano::Face>> Flyscene::firstBox(Tucano::Mesh mesh) {
    std::vector<Tucano::Face> box;
    std::vector<std::vector<Tucano::Face>> boxes;
    for (int i = 0; i < mesh.getNumberOfFaces(); i++) {
        box.push_back(mesh.getFace(i));
    }
    //std::cout << "big box" << box.size() << std::endl;
    //std::cout << "total faces" << mesh.getNumberOfFaces() << std::endl;
    createboxes(box, mesh, boxes);
    return boxes;
}

// checks intersection with bounding box
// src =  slights
std::vector<HitData> intersectBox(Eigen::Vector3f start, Eigen::Vector3f to, std::vector<std::vector<Tucano::Face>> boxes,
                  std::vector<std::vector<Eigen::Vector3f>> boxbounds, int inter_node, Tucano::Mesh mesh, std::vector<HitData>& hits) {
	for (inter_node; inter_node < boxes.size(); ++inter_node) {

		if (boxes.size() <= inter_node) {
			return hits;
		}

		if (boxes[inter_node].size() == 0) {
			continue;
		}

		Eigen::Vector3f box_min = boxbounds[inter_node][0];
		Eigen::Vector3f box_max = boxbounds[inter_node][1];

		float tmin_x = (box_min.x() - start.x()) / to.x();
		float tmax_x = (box_max.x() - start.x()) / to.x();

		float tmin_y = (box_min.y() - start.y()) / to.y();
		float tmax_y = (box_max.y() - start.y()) / to.y();

		float tmin_z = (box_min.z() - start.z()) / to.z();
		float tmax_z = (box_max.z() - start.z()) / to.z();

		float tin_x = std::min(tmin_x, tmax_x);
		float tout_x = std::max(tmin_x, tmax_x);

		float tin_y = std::min(tmin_y, tmax_y);
		float tout_y = std::max(tmin_y, tmax_y);

		float tin_z = std::min(tmin_z, tmax_z);
		float tout_z = std::max(tmin_z, tmax_z);

		float tin = std::max(tin_x, tin_y);
		tin = std::max(tin, tin_z);
		float tout = std::min(tout_x, tout_y);
		tout = std::min(tout, tout_z);

		if ((tin > tout) || tout < 0.000001f) {
			if (boxes.size() <= inter_node) {
				continue;
			}
			//return intersectBox(start, to, boxes, boxbounds, inter_node + 1, mesh, hits);
			continue;
		}

		boxesHit = boxesHit + boxes[inter_node].size();

		for (int i = 0; i < boxes[inter_node].size(); i++) {
			auto ans = intersectTriange(start, to, boxes[inter_node][i], mesh);
			if (ans.inter) {
				HitData hit;
				hit.face = ans.face;
				hit.hit = ans.hit;
				hits.push_back(hit);
			}
		}
	}
	return hits;

}

//INTERSECT RETURNS A STRUCTURE 
// intersect(start, to, mesh).inter -> returns true if there is a intersection
// intersect(start, to, mesh).face -> returns the face that it hit if it didnt hit returns a random face
// intersect(start, to, mesh).hit -> returns the vec3f that it hit if it didnt hit returns a random hit point
//intersect of one vector to the universe
// FOR THE ACCELERATION YOU NEED TO FEED INTERSECT METHOD WITH THOSE CREATE THEM BEFORE LOOPING THEY ARE EXPENSIVE
//std::vector<std::vector<Tucano::Face>> boxes = firstBox(mesh);
//std::vector<std::vector<Eigen::Vector3f>> boxbounds;
//for (int i = 0; i < boxes.size(); i++) {
//boxbounds.push_back(getBoxLimits(boxes[i], mesh));
//}
auto Flyscene::intersect(Eigen::Vector3f start, Eigen::Vector3f to, Tucano::Mesh mesh,
                         std::vector<std::vector<Tucano::Face>> boxes,
                         std::vector<std::vector<Eigen::Vector3f>> boxbounds, Eigen::Vector3f center,
                         float sphere_r) {
    struct result {
        bool inter;
        Tucano::Face face;
        Eigen::Vector3f hit;
    };
    std::vector<HitData> lolz;
    std::vector<HitData> hits;


    hits = intersectBox(start, to, boxes, boxbounds, 0, mesh, lolz);
	interceptions = interceptions + hits.size();

    if (hits.size() == 0) {
        return result{false, mesh.getFace(0), Eigen::Vector3f(Eigen::Vector3f(0, 0, 0))};
    }


    // go through all the hits and get the smallest distance
    Eigen::Vector3f ahit = hits[0].gethit();
    float min_distance = (ahit - start).squaredNorm();
    float the_one = 0;

    for (int i = 1; i < hits.size(); i++) {
        float d = (hits[i].gethit() - start).squaredNorm();

        if (d < min_distance) {
            the_one = i;
            min_distance = d;
        }
    }

    return result{true, hits[the_one].face, hits[the_one].gethit()};
}

// src = raytracing slight
Eigen::Vector3f refract(Eigen::Vector3f v, Eigen::Vector3f normal, float n1, float n2) {
    Eigen::Vector3f vnor = v.normalized();
    Eigen::Vector3f nnor = normal.normalized();
    float cosangle = vnor.dot(nnor);
    float snel = n1 / n2;
    float dot = v.dot(normal);

    return snel * (v - dot * normal) - normal * sqrt(1 - pow(snel, 2) * (1 - pow(dot, 2)));
}

Eigen::Vector3f reflect(const Eigen::Vector3f I, const Eigen::Vector3f N) {
    return I - 2 * I.dot(N) * N;
}




Eigen::Vector3f Flyscene::shade(int level, Eigen::Vector3f hit, Eigen::Vector3f from, Tucano::Face face, Tucano::Mesh mesh,
                                Tucano::Effects::PhongMaterial phong, std::vector<Eigen::Vector4f> lights,
                                std::vector<std::vector<Tucano::Face>> boxes, std::vector<std::vector<Eigen::Vector3f>> boxbounds,
                                Eigen::Vector3f light_intensity, Eigen::Vector3f center, float  sphere_r, bool allowSoftShadows) {


    int finesse = 20; // Finesse of soft shadows. the higher the better detail and lower performance (quadratically)


    Eigen::Vector3f normal3 = face.normal.normalized();

    /// 2) compute eye direction
    int material_id = face.material_id;
    Eigen::Vector3f eye_vec3 = (-from).normalized();

    Eigen::Vector3f ka;
    Eigen::Vector3f kd;
    Eigen::Vector3f ks;
    float n;

    if (material_id < 0) {
        printf("--no-material\n");
        ka = Eigen::Vector3f(.2, .2, .2);
        kd = Eigen::Vector3f(.4, .4, .4);
        ks = Eigen::Vector3f(.8, .8, .8);
        n = 10;
    }
    else {
        eye_vec3 = (-from).normalized();
        ka = phong.getMaterial(material_id).getAmbient();
        kd = phong.getMaterial(material_id).getDiffuse();
        ks = phong.getMaterial(material_id).getSpecular();
        n = phong.getMaterial(material_id).getShininess();
    }

    Eigen::Vector3f color = Eigen::Vector3f(0, 0, 0);

    for (int i = 0; i < lights.size(); i++) {

        Eigen::Vector3f normalToLight = (lights[i].head<3>() - hit).normalized();

        /// 0) compute the light direction

        /// generate points on spehere light source
        std::vector<Eigen::Vector3f> lightPoints;
        lightPoints.push_back(lights[i].head<3>());
        if (allowSoftShadows) {
            //orthogonal bases X and Y oriented orthogonal to normalToLight
            Eigen::Vector3f baseX(0, -lights[i].z(), lights[i].y());
            Eigen::Vector3f baseY(lights[i].y() * lights[i].y() + lights[i].z() * lights[i].z(),-lights[i].x() * lights[i].y(), -lights[i].x() * lights[i].z());

            //using Phyllotaxis to distribute rays around center
            int count = pow(finesse * lights[i].w(), 2);
            float scale = 1.0 * lights[i].w() / count;
            for (int j = 1; j <= count; j++) {
                float r = scale * j;
                float a = j * 2.3911;
                float x = r * cos(a);
                float y = r * sin(a);
                lightPoints.push_back(lights[i].head<3>() + x * baseX + y * baseY);
            }
        }


        // number of rays which hit the source light sucessfully
        int hitCount = 0;

        /// count number of rays hitting
        for(auto lp : lightPoints) {
            Eigen::Vector3f lightDir = (lp - hit).normalized();
            auto intersection = intersect(hit+0.001*lightDir, lightDir, mesh, boxes, boxbounds, center, sphere_r);
            //std::cout << "INTERSECTTTTTTTT " << intersection.inter << std::endl;
            if (!intersection.inter) {
                hitCount += 1;

            }else{

            }
        }


        if(hitCount > 0){
            /// 3) compute ambient, diffuse and specular components
            Eigen::Vector3f A = light_intensity.array() * ka.array();

            float cosss = normal3.dot(normalToLight);
            Eigen::Vector3f D = light_intensity.array() * kd.array() * cosss;

            // reflected https://stackoverflow.com/questions/24132774/trouble-with-phong-shading

            Eigen::Vector3f r = (2 * cosss * normal3) - normalToLight;
            r = r.normalized();
            float coss = eye_vec3.dot(r);
            Eigen::Vector3f S = light_intensity.array() * ks.array() * pow(std::max(coss, 0.0f), n);
            // max because -shinenes doesnt make sense

            /// 4) compute final color using the Phong Model
            color = color + (A + D + S)*(1.0*hitCount/lightPoints.size());
        }
    }

    Eigen::Vector3f avgColor = color / lights.size();

    //reflection
    Eigen::Vector3f reflection = reflect(from, normal3);
    Eigen::Vector3f reflectioncolor = Eigen::Vector3f(0,0,0);
    //WE NEED A MATERIAL PROPERTY
    if (level > 0) {
        reflectioncolor = recursiveraytracing(level - 1, hit, reflection, mesh, phong, lights, boxes, boxbounds, true, center , sphere_r);
    }

    // refraction
    float air = 1.0;
    float material;
    if (!(material_id < 0)) {
        material = materials[material_id].getOpticalDensity();
    }
    else {
        material = 1.2;
    }


    Eigen::Vector3f refraction = refract(from, normal3, air, material);
    Eigen::Vector3f refractioncolor = Eigen::Vector3f(0, 0, 0);
    if (level > 0) {
        refractioncolor = recursiveraytracing(level - 1, hit, refraction, mesh, phong, lights, boxes, boxbounds, true, center , sphere_r);
    }

    return color + reflectioncolor + refractioncolor;
}

Eigen::Vector3f Flyscene::recursiveraytracing(int level, Eigen::Vector3f start, Eigen::Vector3f to, Tucano::Mesh mesh,
                                              Tucano::Effects::PhongMaterial phong, std::vector<Eigen::Vector4f> lights,
                                              std::vector<std::vector<Tucano::Face>> boxes,
                                              std::vector<std::vector<Eigen::Vector3f>> boxbounds, bool ref,
                                              Eigen::Vector3f center, float sphere_r) {

    auto intersection = intersect(start, to, mesh, boxes, boxbounds, center, sphere_r);

    if (!intersection.inter && !ref) {
        return Eigen::Vector3f(1, 1, 1);
    }
    if (!intersection.inter && ref) {
        return Eigen::Vector3f(0, 0, 0);
    }

    return shade(level, intersection.hit, intersection.hit - start, intersection.face,
                 mesh, phong, lights, boxes, boxbounds, Eigen::Vector3f(1, 1, 1), center,
                 sphere_r, true);
}


Eigen::Vector3f
Flyscene::traceRay(Eigen::Vector3f &origin, Eigen::Vector3f &dest, std::vector<std::vector<Tucano::Face>> &boxes,
                   std::vector<std::vector<Eigen::Vector3f>> &boxbounds, Eigen::Vector3f center,
                   float sphere_r) {
    
	return recursiveraytracing(levelReflection, origin, dest - origin, mesh, phong, lights, boxes,
                                                 boxbounds, false, center, sphere_r);   //bounce one more ray
}


void Flyscene::createDebugRay(const Eigen::Vector2f &mouse_pos, Eigen::Vector3f start, Eigen::Vector3f to, int n, int max) {

	if (n >= max) {
		return;
	}


	if (n == 0) {
		ray.resetModelMatrix();
		ray.setSize(0.005, 10.0);

		// from pixel position to world coordinates
		Eigen::Vector3f screen_pos = flycamera.screenToWorld(mouse_pos);

		// direction from camera center to click position
		Eigen::Vector3f dir = (screen_pos - flycamera.getCenter()).normalized();

		// position and orient the cylinder representing the ray
		ray.setOriginOrientation(flycamera.getCenter(), dir);

		for (int i = 0; i < reflection_rays.size(); i++) {
			reflection_rays[i].resetModelMatrix();
			refraction_rays[i].resetModelMatrix();
		}

		reflection_rays.clear();
		refraction_rays.clear();

		start = screen_pos;
		to = dir;
	}

	reflection_rays.push_back(Tucano::Shapes::Cylinder(0.0, 10.0, 16, 64));
	refraction_rays.push_back(Tucano::Shapes::Cylinder(0.0, 10.0, 16, 64));
	reflection_rays[n].setSize(0, 0);
	refraction_rays[n].setSize(0, 0);

	refraction_rays[n].resetModelMatrix();
	refraction_rays[n].resetModelMatrix();

	
	for (int i = n + 1; i < refraction_rays.size(); i++) {
		refraction_rays[i].setSize(0.0, 0.0);
		refraction_rays[i].setSize(0.0, 0.0);
	}
	
	Eigen::Vector3f center = mesh.getCentroid();
	float sphere_r = mesh.getBoundingSphereRadius();
	
	auto intersection = intersect(start, to, mesh, boxes, boxbounds, center, sphere_r);
	
	if (intersection.inter) {

		reflection_rays[n].resetModelMatrix();
		refraction_rays[n].resetModelMatrix();

		if (n == 0) {
			ray.setSize(0.005, distance(flycamera.getCenter(), intersection.hit));
		}

		Eigen::Vector3f facenorm = intersection.face.normal.normalized();
		Eigen::Vector3f reflection = reflect(to, facenorm);
		
		float d = distance(start, intersection.hit);
		
		reflection_rays[n].setOriginOrientation(intersection.hit, reflection);
		reflection_rays[n].setSize(0.005, d);
		
		Eigen::Vector3f color3 = shade(levelReflection, intersection.hit, intersection.hit - start, intersection.face, mesh, phong, lights, boxes, boxbounds, Eigen::Vector3f (1,1,1), center, sphere_r,false);
		Eigen::Vector4f color4 = Eigen::Vector4f(color3.x(), color3.y(), color3.z(), 1);

		reflection_rays[n].setColor(color4);

		
		int material_id;
		float material;
		(intersection.face.material_id < 0) ? material = 0.9 : material = materials[intersection.face.material_id].getOpticalDensity();
		float air = 1.0;
		
		Eigen::Vector3f refraction = refract(to, facenorm, air, material);
		float a =  distance(start, intersection.hit);
		

		refraction_rays[n].setSize(0.005, a);
		refraction_rays[n].setOriginOrientation(intersection.hit, refraction);
		refraction_rays[n].setColor(Eigen::Vector4f(0, 1, 0, 0));

		createDebugRay(mouse_pos, intersection.hit, reflection, n + 1, max);
		
	}
	else {
		if (n == 0) {
			ray.setSize(0.005, 10.0);
			return;
		}
		reflection_rays[n].setSize(0, 10);
	}

	// place the camera representation (frustum) on current camera location,
	camerarep.resetModelMatrix();
	camerarep.setModelMatrix(flycamera.getViewMatrix().inverse());

}