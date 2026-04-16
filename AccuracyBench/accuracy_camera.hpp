#pragma once

#include <cmath>

#include <mr-math/math.hpp>

namespace accbench {

using mr::Matr4f;
using mr::Vec3f;
using mr::math::Camera;
using mr::math::Pitch;
using mr::math::Yaw;

// Mirrors mr::graphics::FPSCamera + mr::Scene projection setup (45°, near 0.01, far 1000).
class FpsCamera {
public:
	Camera<float> &cam() { return _cam; }
	const Camera<float> &cam() const { return _cam; }

	// Row-vector math (mr-math): p_clip = p_world * view * proj → view * proj order.
	Matr4f viewProj() const noexcept { return _cam.perspective() * _cam.frustum(); }

	// Yaw/pitch only (world-up); avoids extra roll that fights the basis and looks "swimmy" in preview.
	FpsCamera &turn(Vec3f delta) noexcept {
		delta *= _sensitivity;
		_cam += Yaw(mr::Radiansf(delta.x()));
		_cam += Pitch(mr::Radiansf(delta.y()));
		return *this;
	}

	FpsCamera &move(Vec3f delta) noexcept {
		_cam += delta * _speed;
		return *this;
	}

	void configureProjection(float aspect) {
		auto &pr = _cam.projection();
		pr.distance = 0.01f;
		pr.far = 1000.f;
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
