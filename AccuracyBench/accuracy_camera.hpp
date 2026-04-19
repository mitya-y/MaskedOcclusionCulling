#pragma once

#include <cmath>

#include <mr-math/math.hpp>

namespace accbench {

using mr::Matr4f;
using mr::Vec3f;
using mr::math::Camera;

// Same control math as mr::graphics::FPSCamera (mr-graphics src/camera/camera.hpp):
// Yaw + Pitch + Roll term to keep camera upright relative to world +Y.
class FpsCamera {
public:
	Camera<float> &cam() { return _cam; }
	const Camera<float> &cam() const { return _cam; }

	Matr4f viewProj() const noexcept { return _cam.perspective() * _cam.frustum(); }

	FpsCamera &turn(Vec3f delta) noexcept {
		delta *= _sensitivity;
		_cam += mr::Yaw(mr::Radiansf(delta.x()));
		_cam += mr::Pitch(mr::Radiansf(delta.y()));
		_cam += mr::Roll(-mr::Radiansf(std::acos(_cam.right() & mr::axis::y)) + mr::pi / 2
		    + mr::Radiansf(delta.z()));
		return *this;
	}

	FpsCamera &move(Vec3f delta) noexcept {
		_cam += delta * _speed;
		return *this;
	}

	void configureProjection(float aspect, float nearPlane = 0.01f, float farPlane = 1000.f) {
		auto &pr = _cam.projection();
		pr.distance = nearPlane;
		pr.far = farPlane;
		const float fovRad = mr::Radiansf(mr::Degreesf{45.f}).value();
		pr.width = 2.f * pr.distance * std::tan(fovRad * 0.5f);
		pr.resize(aspect);
	}

	float speed() const noexcept { return _speed; }
	void speed(float s) noexcept { _speed = s; }

	float sensitivity() const noexcept { return _sensitivity; }
	void sensitivity(float s) noexcept { _sensitivity = s; }

private:
	Camera<float> _cam{
	    Vec3f{1.f, 1.f, 1.f},
	    Vec3f{-1.f, -1.f, -1.f},
	    Vec3f{0.f, 1.f, 0.f},
	};
	float _speed = 0.01f;
	float _sensitivity = 1.f;
};

} // namespace accbench
