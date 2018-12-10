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

static std::shared_ptr<Shader> make_shader(const BYTE* src, const uint32_t src_size,
        const std::unordered_map<std::string,
        std::string>& rename=std::unordered_map<std::string, std::string>()) {
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


struct RenderInfo {
	uint32_t id;
	BufferHash index_buffer, vertex_buffer;
	ShaderHash vertex_shader, pixel_shader;
	TextureHash texture;
};
TOJSON(RenderInfo, index_buffer, vertex_buffer, vertex_shader, pixel_shader, texture);


struct GTA5 : public GameController {
	enum RenderPassType {
		START = 0,
		MAIN = 1,
		END = 2,
	};

	struct VSInfo {
		enum ObjectType {
			UNKNOWN = 0,
			VEHICLE = 1,
			WHEEL = 2,
			TREE = 3,
			PEDESTRIAN = 4,
			BONE_MTX = 5,
			PLAYER = 6,
		};

		ObjectType type = UNKNOWN;

		CBufferLocation rage_matrices;
		CBufferLocation wheel_matrices;
		CBufferLocation rage_bonemtx;

		std::shared_ptr<Shader> dflt;
		std::shared_ptr<Shader> ovrride;
	};

	struct PSInfo {
		bool is_final = false;
		std::shared_ptr<Shader> dflt;
		std::shared_ptr<Shader> ovrride;
	};

	GTA5() : GameController() { }

	virtual bool keyDown(unsigned char key, unsigned char special_status) {
		return false;
	}

	virtual void onInitialize() {
		// Add all copy targets
		addTarget("albedo");
		addTarget("final");
		addTarget("prev_disp", TargetType::R32_FLOAT, true);

		// Add all new targets
		addCustomTarget("flow_disp", TargetType::R32G32B32A32_FLOAT, true);
		addCustomTarget("flow", TargetType::R32G32_FLOAT);
		addCustomTarget("velocity", TargetType::R32G32_FLOAT);
		addCustomTarget("disparity", TargetType::R32_FLOAT);
		addCustomTarget("occlusion", TargetType::R32_FLOAT);
		addCustomTarget("object_id", TargetType::R32_UINT);
	}

	std::unordered_map<ShaderHash, VSInfo> vs_info;
	std::unordered_map<ShaderHash, PSInfo> ps_info;
	std::vector<RenderInfo> render_info;
	VSInfo current_vs;
	PSInfo current_ps;

	std::shared_ptr<Shader> vs_static_shader = make_shader(VS_STATIC, sizeof(VS_STATIC));

    // BRADY
	// std::shared_ptr<Shader> ps_output_shader = make_shader(PS_OUTPUT, sizeof(PS_OUTPUT), {
    //         {"SV_Target5", "flow_disp"},
    //         {"SV_Target6", "object_id"}});
	std::shared_ptr<Shader> ps_output_shader = make_shader(PS_OUTPUT, sizeof(PS_OUTPUT), {
            {"SV_Target6", "flow_disp"},
            {"SV_Target7", "object_id"}});

	std::shared_ptr<Shader> flow_shader = make_shader(PS_FLOW, sizeof(PS_FLOW), {
            {"SV_Target0", "flow"},
            {"SV_Target1", "disparity"},
            {"SV_Target2", "occlusion"},
            {"SV_Target3", "velocity"}});
	std::shared_ptr<Shader> noflow_shader = make_shader(PS_NOFLOW, sizeof(PS_NOFLOW), {
            {"SV_Target0", "flow"},
            {"SV_Target1", "disparity"},
            {"SV_Target2", "occlusion"}});

	std::unordered_set<ShaderHash> int_position = {
		ShaderHash("d05510b7:0d9c59d0:612cd23a:f75d5ebd"),
		ShaderHash("c59148c8:e2f1a5ad:649bb9c7:30454a34"),
		ShaderHash("8a028f64:80694472:4d55d5dd:14c329f2"),
		ShaderHash("53a6e156:ff48c806:433fc787:c150b034"),
		ShaderHash("86de7f78:3ecdef51:f1c6f9e3:c5e6338f"),
		ShaderHash("f37799c8:b304710d:3296b36c:46ea12d4"),
		ShaderHash("4b031811:b6bf1c7f:ef4cd0c1:56541537") };

	virtual void onCreateShader(std::shared_ptr<Shader> shader) override {
		if (shader->type() == Shader::VERTEX) {
			VSInfo info;
			info.rage_matrices = CBufferLocation::scan(shader, "rage_matrices", "gWorld", 4 * 16 * sizeof(float));
			if (info.rage_matrices) {
				info.wheel_matrices = CBufferLocation::scan(shader, "matWheelBuffer", "matWheelWorld", 32 * sizeof(float));
				info.rage_bonemtx   = CBufferLocation::scan(shader, "rage_bonemtx", "gBoneMtx", BONE_MTX_SIZE);
				info.dflt = shader;

				if (info.wheel_matrices)
					info.type = VSInfo::WHEEL;
				else if (hasCBuffer(shader, "vehicle_globals") && hasCBuffer(shader, "vehicle_damage_locals"))
					info.type = VSInfo::VEHICLE;
				else if (hasCBuffer(shader, "trees_common_locals"))
					info.type = VSInfo::TREE;
				else if (hasCBuffer(shader, "ped_common_shared_locals"))
					info.type = VSInfo::PEDESTRIAN;
				else if (info.rage_bonemtx)
					info.type = VSInfo::BONE_MTX;

				bool can_inject = true;
				for (const auto & b : shader->cbuffers())
					if (b.bind_point == 0)
						can_inject = false;
				if (int_position.count(shader->hash()))
					can_inject = false;
				if (can_inject) {
					// Duplicate the shader and copy rage matrices
					auto r = shader->subset({ "SV_Position" });
					r->renameOutput("SV_Position", "PREV_POSITION");
					r->renameCBuffer("rage_matrices", "prev_rage_matrices");
					if (info.wheel_matrices)
						r->renameCBuffer("matWheelBuffer", "prev_matWheelBuffer", 5);
					if (info.rage_bonemtx)
						r->renameCBuffer("rage_bonemtx", "prev_rage_bonemtx", 5);
					info.ovrride = shader->append(r);
					vs_info[shader->hash()] = info;
				}
			}
		}
		if (shader->type() == Shader::PIXEL) {
			PSInfo info;
			info.dflt = shader;

			// prior to v1.0.1365.1
			if (hasTexture(shader, "BackBufferTexture")) {
				info.is_final = true;
				ps_info[shader->hash()] = info;
			}
			// v1.0.1365.1 and newer
			if (hasTexture(shader, "SSLRSampler") && hasTexture(shader, "HDRSampler")) {
				// Other candidate textures include "MotionBlurSampler", "BlurSampler", but might depend on graphics settings
				info.is_final = true;
				ps_info[shader->hash()] = info;
			}
			if (hasCBuffer(shader, "misc_globals")) {
				// Inject the shader output
				info.ovrride = shader->append(ps_output_shader);
				ps_info[shader->hash()] = info;
			}
		}
	}
	bool do_override = false;
	virtual void onBindShader(std::shared_ptr<Shader> shader) {
		if (shader->type() == Shader::VERTEX) {
			auto i = vs_info.find(shader->hash());
			if (i != vs_info.end()) {
				current_vs = i->second;
				if (do_override && current_vs.ovrride)
					bindShader(current_vs.ovrride);
			} else {
				current_vs = VSInfo();
			}
		}
		if (shader->type() == Shader::PIXEL) {
			auto i = ps_info.find(shader->hash());
			if (i != ps_info.end()) {
				current_ps = i->second;
				if (do_override && current_ps.ovrride)
					bindShader(current_ps.ovrride);
			}
			else {
				current_ps = PSInfo();
			}
		}
	}
	virtual void overrideShader() {
		if (!do_override) {
			do_override = true;
			if (current_ps.ovrride)
				bindShader(current_ps.ovrride);
			if (current_vs.ovrride)
				bindShader(current_vs.ovrride);
		}
	}
	virtual void defaultShader() {
		if (do_override) {
			do_override = false;
			if (current_ps.ovrride && current_ps.dflt)
				bindShader(current_ps.dflt);
			if (current_vs.ovrride && current_vs.dflt)
				bindShader(current_vs.dflt);
		}
	}
	virtual void onPostProcess(uint32_t frame_id) override {
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

		// The camera params either change,
		// or there is a certain degree of numerical instability
		// (and fov changes, but setting the camera properly is hard ;( )
		// float disp_ab[2] = { -d / e, -c / d};

		float disp_ab[2] = { 6., 4e-5 };

		disparity_correction->set((const float*)disp_ab, 2, 0 * sizeof(float));
		bindCBuffer(disparity_correction);

		float4x4 velocity_matrix[4] = {
			this->prev_view_proj,
			this->prev_view_proj_inv,
			this->cur_view_proj,
			this->cur_view_proj_inv
		};

		velocity_matrix_buffer->set((const float4x4*)velocity_matrix, 4, 0);
		bindCBuffer(velocity_matrix_buffer);

		if (currentRecordingType() == DRAW)
			callPostFx(flow_shader);
		else
			callPostFx(noflow_shader);
	}

	bool requires_view_proj_recompute = true;

	float4x4 avg_world = 0;
	float4x4 avg_world_view = 0;
	float4x4 avg_world_view_proj = 0;

	float4x4 cur_view_proj = 0;
	float4x4 cur_view_proj_inv = 0;

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

	virtual void onBeginFrame(uint32_t frame_id) override {
		start_time = time();
		TS = 0;

		main_render_pass = RenderPassType::START;
		albedo_output = RenderTargetView();

		if (!id_buffer) id_buffer = createCBuffer("IDBuffer", sizeof(int));
		if (!prev_buffer) prev_buffer = createCBuffer("prev_rage_matrices", 4 * sizeof(float4x4));
		if (!prev_wheel_buffer) prev_wheel_buffer = createCBuffer("prev_matWheelBuffer", 4 * sizeof(float4x4));
		if (!prev_rage_bonemtx) prev_rage_bonemtx = createCBuffer("prev_rage_bonemtx", BONE_MTX_SIZE);
		if (!disparity_correction) disparity_correction = createCBuffer("disparity_correction", 2 * sizeof(float));
		if (!velocity_matrix_buffer) velocity_matrix_buffer = createCBuffer("velocity_matrix_buffer", 4 * sizeof(float4x4));

		last_vehicle.reset();
		wheel_count = 0;
		tracker = trackNextFrame();

		avg_world = 0;
		avg_world_view = 0;
		avg_world_view_proj = 0;

		requires_view_proj_recompute = true;

		TS = 0;

		defaultShader();
		render_info.clear();
	}

	virtual void onEndFrame(uint32_t frame_id) override {
		if (currentRecordingType() == NONE)
			return;

		mul(&prev_view, avg_world.affine_inv(), avg_world_view);
		mul(&prev_view_proj, avg_world.affine_inv(), avg_world_view_proj);

		this->prev_view_proj_inv = this->prev_view_proj.inv();

		current_frame_id++;

		// Copy the disparity buffer for occlusion testing
		copyTarget("prev_disp", "disparity");

		LOG(INFO) << "Elapsed (ms): " << time() - start_time << "\tSize (bytes): " << TS;
	}

	RenderTargetView albedo_output;

	virtual void onBeginDraw(const DrawInfo & info) override {
		if ((currentRecordingType() != NONE) &&
			info.target.outputs.size() &&
			info.target.outputs[0].W == defaultWidth() && info.target.outputs[0].H == defaultHeight() &&
			info.target.outputs.size() >= 2 &&
			info.type == DrawInfo::INDEX && info.instances == 0) {

			if (current_vs.rage_matrices && main_render_pass != RenderPassType::END) {
				std::shared_ptr<GPUMemory> wp = current_vs.rage_matrices.fetch(this, info.buffer.vertex_constant, true);

				// Starting the main render pass
				if (main_render_pass == RenderPassType::START) {
					albedo_output = info.target.outputs[0];
					main_render_pass = RenderPassType::MAIN;
					overrideShader();
				}

				if (main_render_pass == RenderPassType::MAIN) {
					uint32_t id = 0;

					// Fetch the rage matrices gWorld, gWorldView, gWorldViewProj
					if (wp && wp->size() >= 3 * sizeof(float4x4)) {
						const float4x4 * rage_mat = (const float4x4 *)wp->data();
						const float4x4& world = rage_mat[0];

						if (this->requires_view_proj_recompute) {
							mul(&this->cur_view_proj, world.affine_inv(), rage_mat[2]);
							this->cur_view_proj_inv = this->cur_view_proj.inv();

							this->requires_view_proj_recompute = false;
						}

						float4x4 prev_rage[4] = { rage_mat[0], rage_mat[1], rage_mat[2], rage_mat[3] };

						mul(&prev_rage[1], rage_mat[0], prev_view);
						mul(&prev_rage[2], rage_mat[0], prev_view_proj);

						// Sum up the world and world_view_proj matrices to later compute the view_proj matrix
						if (current_vs.type != VSInfo::PEDESTRIAN && current_vs.type != VSInfo::PLAYER) {
							// There is a 'BUG' (or feature) in GTA V that doesn't draw Franklyn correctly
							// in first person view (rage_mat are wrong)
							add(&avg_world, avg_world, rage_mat[0]);
							add(&avg_world_view, avg_world_view, rage_mat[1]);
							add(&avg_world_view_proj, avg_world_view_proj, rage_mat[2]);
						}

						if (current_vs.type == VSInfo::WHEEL && last_vehicle) {
							std::shared_ptr<GPUMemory> wm = current_vs.wheel_matrices.fetch(this, info.buffer.vertex_constant, true);

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
							// Determine the GTA type for search
							TrackedFrame::ObjectType gta_type = TrackedFrame::UNKNOWN;

							if (current_vs.type == VSInfo::PEDESTRIAN)
								gta_type = TrackedFrame::PED;
							if (current_vs.type == VSInfo::VEHICLE)
								gta_type = TrackedFrame::VEHICLE;

							Vec3f v = { rage_mat[0].d[3][0], rage_mat[0].d[3][1], rage_mat[0].d[3][2] };
							Quaternion orientation = Quaternion::fromMatrix(world);

							TrackedFrame::Object * object;

							// A tighter radius for unknown objects
							if (gta_type == TrackedFrame::UNKNOWN)
								object = (*tracker)(v, orientation, 0.01f, 0.01f, gta_type);
							else if (gta_type == TrackedFrame::PED)
								object = (*tracker)(v, orientation, 1.f, 10.f, gta_type);
							else
								object = (*tracker)(v, orientation, 0.1f, 0.1f, TrackedFrame::UNKNOWN);

							if (object) {
								std::shared_ptr<TrackData> track = std::dynamic_pointer_cast<TrackData>(object->private_data);
								if (!track) // Create a track if the object is new
									object->private_data = track = std::make_shared<TrackData>();

								// Advance a tracked frame
								if (track->last_frame < current_frame_id) {
									// Advance time
									track->swap();

									// Update the rage matrix
									memcpy(track->cur_rage, rage_mat, sizeof(track->cur_rage));
									track->has_cur_rage = 1;

									if (track->last_frame != current_frame_id - 1) {
										// Avoid objects poping in and out
										track->has_prev_rage = 0;
										track->prev_bones.clear();
										track->prev_wheels.clear();
									}

									track->last_frame = current_frame_id;
								}

								// Update the bone_mtx
								if (current_vs.rage_bonemtx) {
									std::shared_ptr<GPUMemory> bm = current_vs.rage_bonemtx.fetch(this, info.buffer.vertex_constant, true);
									if (bm) {
										if (!track->cur_bones.count(info.buffer.vertex.id)) {
											memcpy(&track->cur_bones[info.buffer.vertex.id], bm->data(), sizeof(TrackData::BoneData));
											TS += sizeof(TrackData::BoneData);
										}
										else if (0) {
											if (memcmp(&track->cur_bones[info.buffer.vertex.id], bm->data(), sizeof(TrackData::BoneData))) {
												LOG(WARN) << "Bone matrix changed for object " << info.shader.pixel << " " << info.shader.vertex << " " << info.buffer.vertex.id;
												LOG(INFO) << object->type() << " " << object->id;
												hideDraw();
												return;
											}
										}
									}
								}

								// Set the prior projection view
								if (track->has_prev_rage)
									memcpy(prev_rage, &track->prev_rage, sizeof(prev_rage));

								// Set the prior bone mtx
								if (current_vs.rage_bonemtx) {
									if (track->prev_bones.count(info.buffer.vertex.id))
										prev_rage_bonemtx->set(track->prev_bones[info.buffer.vertex.id]);
									else if (track->cur_bones.count(info.buffer.vertex.id))
										prev_rage_bonemtx->set(track->cur_bones[info.buffer.vertex.id]);

									bindCBuffer(prev_rage_bonemtx);
								}

								track->id = id = MAX_UNTRACKED_OBJECT_ID + object->id;

								if (current_vs.type == VSInfo::VEHICLE) {
									last_vehicle = track;
									wheel_count = 0;
								}
							}
							else if (current_vs.type == VSInfo::PEDESTRIAN) {
								// Untracked Ped
								hideDraw();
								return;
							}
						}

						prev_buffer->set((float4x4*)prev_rage, 4, 0);
						bindCBuffer(prev_buffer);

						if (!id) {
							id = render_info.size() + 1;
							render_info.push_back({ id, info.buffer.index_hash, info.buffer.vertex_hash, info.shader.vertex, info.shader.pixel, info.shader.ps_texture_hash });
						}

						id_buffer->set(id);
						bindCBuffer(id_buffer);
					}
				}
			}
		}
		else if (main_render_pass == RenderPassType::MAIN) {
			// End of the main render pass
			copyTarget("albedo", albedo_output);
			main_render_pass = RenderPassType::END;
			defaultShader();
		}
	}

	virtual void onEndDraw(const DrawInfo & info) override {
		if (current_ps.is_final) {
			// Draw the final image (right before the image is distorted)
			copyTarget("final", info.target.outputs[0]);
		}
	}

	virtual std::string provideGameState() const override {
		std::string state = "{ \"render_info\" : " + toJSON(render_info);
		if (tracker)
			return state + "," + toJSON(tracker->info).substr(1);
		return state + "}";
	}

	virtual bool stop() { return stopTracker(); }

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
