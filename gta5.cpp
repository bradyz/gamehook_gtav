#include <windows.h>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <iterator>

#include "scripthook\main.h"
#include "scripthook\natives.h"

#include "log.h"
#include "sdk.h"
#include "util.h"
#include "gtastate.h"
#include "ps_output.h"
#include "vs_static.h"
#include "ps_flow.h"
#include "ps_noflow.h"

#define DOT(a,b,i,j) (a[0][i]*b[0][j] + a[1][i]*b[1][j] + a[2][i]*b[2][j] + a[3][i]*b[3][j])

const uint32_t MAX_UNTRACKED_OBJECT_ID = 1 << 14;
const size_t RAGE_MAT_SIZE = 4 * sizeof(float4x4);
const size_t VEHICLE_SIZE = RAGE_MAT_SIZE;
const size_t WHEEL_SIZE = RAGE_MAT_SIZE + 2 * sizeof(float4x4);
const size_t BONE_MTX_SIZE = 255 * 4 * 3 * sizeof(float);

// For debugging, use the current matrices, not the past to estimate the flow
//#define CURRENT_FLOW

double time() {
	auto now = std::chrono::system_clock::now();
	return std::chrono::duration<double>(now.time_since_epoch()).count();
}

static std::shared_ptr<Shader> make_shader(const BYTE* src, const uint32_t src_size, const std::unordered_map<std::string, std::string>& rename=std::unordered_map<std::string, std::string>()) {
	return Shader::create(ByteCode(src, src + src_size), rename);
}

struct TrackData : public TrackedFrame::PrivateData {
	struct BoneData {
		float data[255][3][4] = { 0 };
	};

	struct WheelData {
		float4x4 data[2] = { 0 };
	};

	uint32_t id = 0;
	uint32_t last_frame = 0;

	// Previous frame data.
	uint32_t has_prev_rage = 0;
	float4x4 prev_rage[4] = { 0 };
	std::unordered_map<int, BoneData> prev_bones;
	std::vector<WheelData> prev_wheels;

	// Current frame data.
	uint32_t has_cur_rage = 0;
	float4x4 cur_rage[4] = { 0 };
	std::unordered_map<int, BoneData> cur_bones;
	std::vector<WheelData> cur_wheels;

	void swap() {
		has_prev_rage = has_cur_rage;
		memcpy(prev_rage, cur_rage, sizeof(prev_rage));
		prev_bones.swap(cur_bones);
		prev_wheels.swap(cur_wheels);
	}
};

struct VehicleTrack {
	float4x4 rage[4];
	float4x4 wheel[16][2];
};

struct GTA5 : public GameController {
	enum ObjectType {
		UNKNOWN = 0,
		VEHICLE = 1,
		WHEEL = 2,
		TREE = 3,
		PEDESTRIAN = 4,
		BONE_MTX = 5,
		PLAYER = 6,
	};

	enum RenderPassType {
		END = 0,
		START = 1,
		MAIN = 2,
	};

	CBufferVariable rage_matrices = { "rage_matrices", "gWorld",{ 0 },{ 4 * 16 * sizeof(float) } };
	CBufferVariable wheel_matrices = { "matWheelBuffer", "matWheelWorld",{ 0 },{ 32 * sizeof(float) } };
	CBufferVariable rage_bonemtx = { "rage_bonemtx", "gBoneMtx",{ 0 },{ BONE_MTX_SIZE } };

	std::unordered_map<ShaderHash, ObjectType> object_type;

	// Find a better way to find the final shader.
	std::unordered_set<ShaderHash> final_shader;
	// std::unordered_set<ShaderHash> final_shader = { ShaderHash("6eef7895:a8818cdd:d19840f5:68c224f4") };

	std::shared_ptr<Shader> vs_static_shader = make_shader(VS_STATIC, sizeof(VS_STATIC));
	std::shared_ptr<Shader> ps_output_shader = make_shader(PS_OUTPUT, sizeof(PS_OUTPUT), { {"SV_Target5", "flow_disp"}, {"SV_Target6", "object_id"} });
	std::shared_ptr<Shader> flow_shader = make_shader(PS_FLOW, sizeof(PS_FLOW), { {"SV_Target0", "flow"}, {"SV_Target1", "disparity"}, {"SV_Target2", "occlusion"}, {"SV_Target3", "velocity"} });
	std::shared_ptr<Shader> noflow_shader = make_shader(PS_NOFLOW, sizeof(PS_NOFLOW), { {"SV_Target0", "flow"}, {"SV_Target1", "disparity"}, {"SV_Target2", "occlusion"} });

	std::unordered_set<ShaderHash> int_position = {
		ShaderHash("d05510b7:0d9c59d0:612cd23a:f75d5ebd"),
		ShaderHash("c59148c8:e2f1a5ad:649bb9c7:30454a34"),
		ShaderHash("8a028f64:80694472:4d55d5dd:14c329f2"),
		ShaderHash("53a6e156:ff48c806:433fc787:c150b034"),
		ShaderHash("86de7f78:3ecdef51:f1c6f9e3:c5e6338f"),
		ShaderHash("f37799c8:b304710d:3296b36c:46ea12d4"),
		ShaderHash("4b031811:b6bf1c7f:ef4cd0c1:56541537") };

	GTA5() : GameController() {	}

	virtual bool keyDown(unsigned char key, unsigned char special_status) {
		return false;
	}

	virtual std::vector<ProvidedTarget> providedTargets() const override {
		return {
			{ "albedo" },
			{ "final" },
			{ "prev_disp", TargetType::R32_FLOAT, true }
		};
	}

	virtual std::vector<ProvidedTarget> providedCustomTargets() const {
		// Write the disparity into a custom render target (this name needs to match the injection shader buffer name!)
		return {
			{"flow_disp", TargetType::R32G32B32A32_FLOAT, true},
			{ "flow", TargetType::R32G32_FLOAT },
			{ "disparity", TargetType::R32_FLOAT },
			{ "occlusion", TargetType::R32_FLOAT },
			{ "object_id", TargetType::R32_UINT },
			{ "velocity", TargetType::R32G32B32A32_FLOAT },
		};
	}

	virtual std::shared_ptr<Shader> injectShader(std::shared_ptr<Shader> shader) {
		if (shader->type() == Shader::VERTEX) {
			if (!rage_matrices.scan(shader))
				return nullptr;

			ObjectType ot = object_type[shader->hash()] = this->findObjectType(shader);

			if (!this->canInject(shader))
				return nullptr;

			// Duplicate the shader and copy rage matrices
			std::shared_ptr<Shader> modified_shader = shader->subset({ "SV_Position" });

			modified_shader->renameOutput("SV_Position", "PREV_POSITION");
			modified_shader->renameCBuffer("rage_matrices", "prev_rage_matrices");

			if (ot == WHEEL)
				modified_shader->renameCBuffer("matWheelBuffer", "prev_matWheelBuffer", 5);
			else if (ot == PEDESTRIAN || ot == BONE_MTX)
				modified_shader->renameCBuffer("rage_bonemtx", "prev_rage_bonemtx", 5);

			// TODO: Handle characters properly
			// return vs_static_shader;

			return modified_shader;
		}
		else if (shader->type() == Shader::PIXEL) {
			if (hasTexture(shader, "BackBufferTexture")) {
				final_shader.insert(shader->hash());
			}

			// v1.0.1365.1 and newer
			if (hasTexture(shader, "SSLRSampler") && hasTexture(shader, "HDRSampler")) {
				// Other candidate textures include "MotionBlurSampler", "BlurSampler", but might depend on graphics settings
				final_shader.insert(shader->hash());
			}

			if (hasCBuffer(shader, "misc_globals"))
				return ps_output_shader;
		}

		return nullptr;
	}

	virtual void postProcess(uint32_t frame_id) override {
		if (currentRecordingType() == NONE)
			return;

		// Estimate the projection matrix (or at least a subset of it's values)
		// a 0 0 0
		// 0 b 0 0
		// 0 0 c e
		// 0 0 d 0
		// With a bit of math we can see that avg_world_view_proj[:][3] = d * avg_world_view[:][2], below is the least squares estimate for d
		// and avg_world_view_proj[:][2] = c * avg_world_view[:][2] + e * avg_world_view[:][3], below is the least squares estimate for c and e
		// float d = DOT(avg_world_view_proj, avg_world_view, 3, 2) / DOT(avg_world_view, avg_world_view, 2, 2);
		// float A00 = DOT(avg_world_view, avg_world_view, 2, 2), A10 = DOT(avg_world_view, avg_world_view, 2, 3), A11 = DOT(avg_world_view, avg_world_view, 3, 3);
		// float b0 = DOT(avg_world_view, avg_world_view_proj, 2, 2), b1 = DOT(avg_world_view, avg_world_view_proj, 3, 2);
		//
		// float det = 1.f / (A00*A11 - A10 * A10);
		// float c = det * (b0 * A11 - b1 * A10);
		// float e = det * (b1 * A00 - b0 * A10);
		//
		// float disp_ab[2] = { -d / e, -c / d };
		// LOG(INFO) << "disp_ab " << -d / e << " " << -c / d;

		// The camera params either change, or there is a certain degree of numerical instability (and fov changes, but setting the camera properly is hard ;( )
		float disp_ab[2] = { 6., 4e-5 };

		disparity_correction->set((const float*)disp_ab, 2, 0 * sizeof(float));
		bindCBuffer(disparity_correction);

		if (currentRecordingType() == DRAW)
			callPostFx(flow_shader);
		else
			callPostFx(noflow_shader);
	}

	bool needs_mat_recompute = false;

	float4x4 cur_view = 0;
	float4x4 cur_view_proj = 0;
	float4x4 cur_view_proj_inv = 0;

	float4x4 avg_world = 0;
	float4x4 avg_world_view = 0;
	float4x4 avg_world_view_proj = 0;

	float4x4 prev_view = 0;
	float4x4 prev_view_proj = 0;
	float4x4 prev_view_proj_inv = 0;

	RenderPassType main_render_pass = RenderPassType::END;

	std::shared_ptr<CBuffer> id_buffer;
	std::shared_ptr<CBuffer> prev_buffer;
	std::shared_ptr<CBuffer> prev_wheel_buffer;
	std::shared_ptr<CBuffer> prev_rage_bonemtx;
	std::shared_ptr<CBuffer> disparity_correction;

	std::shared_ptr<CBuffer> velocity_matrix_buffer;

	TrackedFrame * tracker = nullptr;
	std::shared_ptr<TrackData> last_vehicle;
	uint32_t wheel_count = 0;

	double start_time;
	size_t TS = 0;
	uint32_t current_frame_id = 1;

	virtual void startFrame(uint32_t frame_id) override {
		start_time = time();
		TS = 0;

		main_render_pass = RenderPassType::START;
		albedo_output = RenderTargetView();

		if (!id_buffer) id_buffer = createCBuffer("IDBuffer", sizeof(int));
		if (!prev_buffer) prev_buffer = createCBuffer("prev_rage_matrices", 4 * sizeof(float4x4));
		if (!prev_wheel_buffer) prev_wheel_buffer = createCBuffer("prev_matWheelBuffer", 4 * sizeof(float4x4));
		if (!prev_rage_bonemtx) prev_rage_bonemtx = createCBuffer("prev_rage_bonemtx", BONE_MTX_SIZE);
		if (!disparity_correction) disparity_correction = createCBuffer("disparity_correction", 2 * sizeof(float));

		if (!velocity_matrix_buffer) velocity_matrix_buffer = createCBuffer("velocity_matrix_buffer", 3 * sizeof(float4x4));

		last_vehicle.reset();
		wheel_count = 0;
		tracker = trackNextFrame();

		avg_world = 0;
		avg_world_view = 0;
		avg_world_view_proj = 0;

		needs_mat_recompute = true;
	}

	virtual void endFrame(uint32_t frame_id) override {
		if (currentRecordingType() == NONE)
			return;

		mul(&prev_view_proj, avg_world.affine_inv(), avg_world_view_proj);
		mul(&prev_view, avg_world.affine_inv(), avg_world_view);

		this->prev_view = this->cur_view;
		this->prev_view_proj = this->cur_view_proj;
		this->prev_view_proj_inv = this->cur_view_proj_inv;

		current_frame_id++;

		// Copy the disparity buffer for occlusion testing
		copyTarget("prev_disp", "disparity");

		LOG(INFO) << "Time elapsed (milliseconds): " << time() - start_time << "\tTotal size (bytes): " << TS;
	}

	RenderTargetView albedo_output;

	virtual DrawType startDraw(const DrawInfo &info) override {
		if ((currentRecordingType() != NONE) &&
				info.outputs.size() && info.outputs[0].W == defaultWidth() && info.outputs[0].H == defaultHeight() &&
				info.outputs.size() >= 2 &&
				info.type == DrawInfo::INDEX && info.instances == 0) {

			if (!rage_matrices.has(info.vertex_shader))
				return DEFAULT;
			else if (main_render_pass == RenderPassType::END)
				return DEFAULT;

			ObjectType type = UNKNOWN;
			{
				auto i = object_type.find(info.vertex_shader);
				if (i != object_type.end())
					type = i->second;
			}

			std::shared_ptr<GPUMemory> wp = rage_matrices.fetch(this, info.vertex_shader, info.vs_cbuffers, true);

			// Starting the main render pass
			if (main_render_pass == RenderPassType::START) {
				albedo_output = info.outputs[0];

				main_render_pass = RenderPassType::MAIN;
			}

			// Need at least world, world_view, world_view_proj matrices.
			if (!wp || wp->size() < 3 * sizeof(float4x4))
				return DEFAULT;

			// Contains gWorld, gWorldView, gWorldViewProj, gViewInverse.
			const float4x4* rage_mat = (const float4x4*)wp->data();
			const float4x4& world = rage_mat[0];

			float4x4 prev_rage[4] = { rage_mat[0], rage_mat[1], rage_mat[2], rage_mat[3] };

			// Compute the matrices once per frame.
			if (this->needs_mat_recompute) {
				const float4x4& world_inv = world.affine_inv();

				mul(&this->cur_view, world_inv, rage_mat[1]);
				mul(&this->cur_view_proj, world_inv, rage_mat[2]);

				float4x4 proj;
				
				mul(&proj, rage_mat[1].affine_inv(), rage_mat[2]);

				const float4x4& proj_inv = proj.affine_inv();

				mul(&this->cur_view_proj_inv, proj_inv, rage_mat[3]);

				float4x4 debug_cur_view = rage_mat[3].affine_inv();
				float4x4 debug_view_proj_inv = cur_view_proj.affine_inv();
				float4x4 debug_proj;

				// LOG(INFO) << debug_view_proj_inv;

				cur_view_proj_inv = debug_view_proj_inv;

				float4x4 velocity_matrix[3] = { prev_view_proj_inv, cur_view_proj_inv, cur_view };

				mul(&debug_proj, rage_mat[1].affine_inv(), rage_mat[2]);

				//LOG(INFO) << debug_cur_view;
				//LOG(INFO) << world_inv;
				//LOG(INFO) << debug_proj;
				//LOG(INFO) << cur_view_proj_inv;
				//LOG(INFO) << cur_view;

				velocity_matrix_buffer->set(velocity_matrix);
				bindCBuffer(velocity_matrix_buffer);

				//for (int i = 0; i < 4; i++)
				//	for (int j = 0; j < 4; j++)
				//		if (proj[i][j] > -1e-2 && proj[i][j] < 1e-2)
				//			proj[i][j] = 0.0;

				this->needs_mat_recompute = false;
			}

			mul(&prev_rage[1], world, prev_view);
			mul(&prev_rage[2], world, prev_view_proj);

			uint32_t id = 0;

			// Sum up the world and world_view_proj matrices to later compute the view_proj matrix
			// There is a 'BUG' (or feature) in GTA V that doesn't draw Franklyn correctly in first person view (rage_mat are wrong)
			if (type != PEDESTRIAN && type != PLAYER) {
				add(&avg_world, avg_world, rage_mat[0]);
				add(&avg_world_view, avg_world_view, rage_mat[1]);
				add(&avg_world_view_proj, avg_world_view_proj, rage_mat[2]);
			}

			if (type == WHEEL && last_vehicle) {
				std::shared_ptr<GPUMemory> wm = wheel_matrices.fetch(this, info.vertex_shader, info.vs_cbuffers, true);

				if (wm && wm->size() >= 2 * sizeof(float4x4)) {
					if (last_vehicle->cur_wheels.size() <= wheel_count)
						last_vehicle->cur_wheels.resize(wheel_count + 1);

					memcpy(&last_vehicle->cur_wheels[wheel_count], wm->data(), sizeof(TrackData::WheelData));

					// Set the previous wheel matrix
					if (wheel_count < last_vehicle->prev_wheels.size())
						prev_wheel_buffer->set(last_vehicle->prev_wheels[wheel_count]);
					else
						prev_wheel_buffer->set(last_vehicle->cur_wheels[wheel_count]);

					bindCBuffer(prev_wheel_buffer);
				}

				id = last_vehicle->id;
				wheel_count++;
			}
			else if (tracker) {
				TrackedFrame::ObjectType gta_type = TrackedFrame::UNKNOWN;

				if (type == PEDESTRIAN)
					gta_type = TrackedFrame::PED;
				else if (type == VEHICLE)
					gta_type = TrackedFrame::VEHICLE;

				Vec3f position = { world[3][0], world[3][1], world[3][2] };
				Quaternion orientation = Quaternion::fromMatrix(world);

				TrackedFrame::Object* object;

				// A tighter radius for unknown objects
				if (gta_type == TrackedFrame::UNKNOWN)
					object = (*tracker)(position, orientation, 0.01f, 0.01f, TrackedFrame::UNKNOWN);
				else if (gta_type == TrackedFrame::PED)
					object = (*tracker)(position, orientation, 1.f, 10.f, TrackedFrame::PED);
				else
					object = (*tracker)(position, orientation, 0.1f, 0.1f, TrackedFrame::UNKNOWN);

				if (object) {
					std::shared_ptr<TrackData> track = std::dynamic_pointer_cast<TrackData>(object->private_data);

					// Create a track if the object is new
					if (!track)
						object->private_data = track = std::make_shared<TrackData>();

					if (object->type() == TrackedFrame::PLAYER)
						type = PLAYER;

					// Advance a tracked frame
					if (track->last_frame < current_frame_id) {
						// Advance time
						track->swap();

						// Update the rage matrix
						memcpy(track->cur_rage, rage_mat, sizeof(track->cur_rage));
						track->has_cur_rage = 1;

						// Avoid objects poping in and out
						if (track->last_frame != current_frame_id - 1) {
							track->has_prev_rage = 0;
							track->prev_bones.clear();
							track->prev_wheels.clear();
						}

						track->last_frame = current_frame_id;
					}

					// Update the bone_mtx
					if (type == PEDESTRIAN || type == BONE_MTX) {
						std::shared_ptr<GPUMemory> bm = rage_bonemtx.fetch(this, info.vertex_shader, info.vs_cbuffers, true);

						if (bm) {
							if (!track->cur_bones.count(info.vertex_buffer.id)) {
								memcpy(&track->cur_bones[info.vertex_buffer.id], bm->data(), sizeof(TrackData::BoneData));
								TS += sizeof(TrackData::BoneData);
							}
							else if (0) {
								if (memcmp(&track->cur_bones[info.vertex_buffer.id], bm->data(), sizeof(TrackData::BoneData))) {
									LOG(WARN) << "Bone matrix changed for object " << info.pixel_shader << " " << info.vertex_shader << " " << info.vertex_buffer.id;
									LOG(INFO) << object->type() << " " << object->id;
									return HIDE;
								}
							}
						}
					}

					// Set the prior projection view
					if (track->has_prev_rage)
						memcpy(prev_rage, &track->prev_rage, sizeof(prev_rage));

					// Set the prior bone mtx
					if (type == PEDESTRIAN || type == BONE_MTX) {
						if (track->prev_bones.count(info.vertex_buffer.id))
							prev_rage_bonemtx->set(track->prev_bones[info.vertex_buffer.id]);
						else if (track->cur_bones.count(info.vertex_buffer.id))
							prev_rage_bonemtx->set(track->cur_bones[info.vertex_buffer.id]);

						bindCBuffer(prev_rage_bonemtx);
					}

					track->id = id = MAX_UNTRACKED_OBJECT_ID + object->id;

					if (type == VEHICLE) {
						last_vehicle = track;
						wheel_count = 0;
					}
				}
				else if (type == PEDESTRIAN) {
					return HIDE;
				}
			}

			prev_buffer->set((float4x4*)prev_rage, 4, 0);
			bindCBuffer(prev_buffer);

			id_buffer->set(id);
			bindCBuffer(id_buffer);

			return RIGID;
		}
		else if (main_render_pass == RenderPassType::MAIN) {
			copyTarget("albedo", albedo_output);

			// End of the main render pass
			main_render_pass = RenderPassType::END;
		}

		return DEFAULT;
	}

	virtual void endDraw(const DrawInfo & info) override {
		// Draw the final image (right before the image is distorted)
		if (final_shader.count(info.pixel_shader))
			copyTarget("final", info.outputs[0]);
	}

	virtual std::string gameState() const override {
		if (tracker)
			return toJSON(tracker->info);

		return "";
	}

	virtual bool stop() { return stopTracker(); }

	ObjectType findObjectType(std::shared_ptr<Shader> shader) {
		if (hasCBuffer(shader, "vehicle_globals") && hasCBuffer(shader, "vehicle_damage_locals"))
			return VEHICLE;
		else if (hasCBuffer(shader, "trees_common_locals"))
			return TREE;
		else if (hasCBuffer(shader, "ped_common_shared_locals"))
			return PEDESTRIAN;
		else if (wheel_matrices.scan(shader))
			return WHEEL;
		else if (rage_bonemtx.scan(shader))
			return BONE_MTX;

		return UNKNOWN;
	}

	bool canInject(std::shared_ptr<Shader> shader) {
		for (const auto & b : shader->cbuffers()) {
			if (b.bind_point == 0)
				return false;
		}

		if (int_position.count(shader->hash()))
			return false;

		return true;
	}
};

REGISTER_CONTROLLER(GTA5);

BOOL WINAPI DllMain(HINSTANCE hInst, DWORD reason, LPVOID) {
	if (reason == DLL_PROCESS_ATTACH) {
		LOG(INFO) << "GTA5 turned on";
		initGTA5State(hInst);
	}

	if (reason == DLL_PROCESS_DETACH) {
		releaseGTA5State(hInst);
		LOG(INFO) << "GTA5 turned off";
	}
	return TRUE;
}