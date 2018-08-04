#include "gtastate.h"
#include "log.h"
#include "scripthook/main.h"
#include "scripthook/types.h"
#include "scripthook/enums.h"
#include "scripthook/natives.h"
#include <string>
#include <ostream>
#include <mutex>
#include <Windows.h>

const float TRACKING_RAD = 0.5f;
const float TRACKING_QUAT = 0.1f;

struct Tracker {
	static std::mutex is_tracking;

	static TrackedFrame current;
	static TrackedFrame returned;

	static uint64_t current_id;
	static uint64_t returned_id;

	static bool stop_tracking;
	static bool currently_tracking;

	static void Main() {
		current_id = 1;
		returned_id = 1;

		stop_tracking = false;
		currently_tracking = true;

		while (!stop_tracking) {
			{
				std::lock_guard<std::mutex> lock(is_tracking);

				current.fetch();
				current_id += 1;
			}

			WAIT(0);
		}

		currently_tracking = false;

		TERMINATE();
	}

	static TrackedFrame* nextFrame() {
		if (!current_id)
			return nullptr;

		std::lock_guard<std::mutex> lock(is_tracking);

		uint64_t delta = current_id - returned_id;

		if (delta > 0) {
			for (int i = 0; i < N_OBJECTS; i++) {
				TrackedFrame::Object& cur = current.objects[i];
				TrackedFrame::Object& ret = returned.objects[i];

				if (ret.id == cur.id) {
					ret.age = cur.age = ret.age + (uint32_t)delta;
					ret.p = cur.p;
					ret.q = cur.q;
				}
				else if (cur.id) {
					// Associate the private data with the returned object only [no swapping here].
					ret = cur;
				}
				else if (ret.id) {
					ret.id = 0;
					ret.private_data.reset();
				}
			}

			returned.object_map.swap(current.object_map);
			returned.info = current.info;
			returned_id = current_id;
		}

		return &returned;
	}

	static bool stop() {
		stop_tracking = true;

		return !currently_tracking;
	}
};

bool stopTracker() {
	return Tracker::stop();
}

std::mutex Tracker::is_tracking;

TrackedFrame Tracker::current;
TrackedFrame Tracker::returned;

uint64_t Tracker::current_id = 0;
uint64_t Tracker::returned_id = 0;

bool Tracker::stop_tracking = false;
bool Tracker::currently_tracking = false;

TrackedFrame* trackNextFrame() {
	return Tracker::nextFrame();
}

void initGTA5State(HMODULE hInstance) {
	scriptRegister(hInstance, Tracker::Main);
}

void releaseGTA5State(HMODULE hInstance) {
	scriptUnregister(hInstance);
}

TrackedFrame::TrackedFrame() :object_map(TRACKING_RAD) {}

// Object type hidden in the 4 rightmost bits.
uint32_t ID(uint32_t id, TrackedFrame::ObjectType t) {
	return ((uint32_t)t) << 28 | id;
}

void TrackedFrame::fetch() {
	static std::mutex fetching;
	std::lock_guard<std::mutex> lock(fetching);

	// Clear the tracker
	object_map.clear();
	for (int i = 0; i < N_OBJECTS; i++)
		objects[i].id = 0;

	// Track all new objects
	this->fetch_helper(&worldGetAllPeds, PED);
	this->fetch_helper(&worldGetAllObjects, OBJECT);
	this->fetch_helper(&worldGetAllPickups, PICKUP);
	this->fetch_helper(&worldGetAllVehicles, VEHICLE);

	Player p = PLAYER::PLAYER_ID();
	Ped pp = PLAYER::PLAYER_PED_ID();

	Vector3 position = ENTITY::GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(pp, 0.0, 0.0, 0.0);
	Vector3 forward_vector = ENTITY::GET_ENTITY_FORWARD_VECTOR(pp);

	info.time_since_player_drove_against_traffic = PLAYER::GET_TIME_SINCE_PLAYER_DROVE_AGAINST_TRAFFIC(p);
	info.time_since_player_drove_on_pavement = PLAYER::GET_TIME_SINCE_PLAYER_DROVE_ON_PAVEMENT(p);
	info.time_since_player_hit_ped = PLAYER::GET_TIME_SINCE_PLAYER_HIT_PED(p);
	info.time_since_player_hit_vehicle = PLAYER::GET_TIME_SINCE_PLAYER_HIT_VEHICLE(p);

	info.dead = PLAYER::IS_PLAYER_DEAD(p);
	info.position = { position.x, position.y, position.z };
	info.forward_vector = { forward_vector.x, forward_vector.y, forward_vector.z };
	info.heading = ENTITY::GET_ENTITY_HEADING(pp);

	info.on_foot = PED::IS_PED_ON_FOOT(pp);
	info.in_vehicle = 2 * PED::IS_PED_GETTING_INTO_A_VEHICLE(pp) + PED::IS_PED_IN_ANY_VEHICLE(pp, true);
	info.on_bike = PED::IS_PED_ON_ANY_BIKE(pp);

	// Let's assume we play with Franklyn all the time
	STATS::STAT_GET_INT(GAMEPLAY::GET_HASH_KEY("SP1_TOTAL_CASH"), &info.money, -1);
}


void TrackedFrame::fetch_helper(int(*object_getter)(int*, int), ObjectType object_type) {
	Player p = PLAYER::PLAYER_ID();
	Ped pp = PLAYER::PLAYER_PED_ID();
	Entity player_ped = PLAYER::PLAYER_PED_ID();

	static int entity_buf[N_ENTITIES];

	int num_objects = object_getter(entity_buf, N_ENTITIES);

	for (int i = 0; i < num_objects; i++) {
		const int e = entity_buf[i];
		Quaternion q;
		ENTITY::GET_ENTITY_QUATERNION(e, &q.x, &q.y, &q.z, &q.w);
		Vector3 p = ENTITY::GET_OFFSET_FROM_ENTITY_IN_WORLD_COORDS(e, 0.0, 0.0, 0.0);

		// Add the entry
		uint32_t k = (e >> 8) & (N_OBJECTS / 2 - 1);

		if (objects[k].id)
			LOG(WARN) << "Tracker has duplicate objects";

		objects[k] = { ID(e, e == player_ped ? ObjectType::PLAYER : object_type), 0, {p.x, p.y, p.z}, q, nullptr };
		object_map.insert({ objects[k].p.x, objects[k].p.y }, k);

		// Track the head gear
		if (object_type == PED) {
			Vector3 hp = PED::GET_PED_BONE_COORDS(e, SKEL_Head, 0.0, 0.0, 0.0);
			uint32_t kk = k + N_OBJECTS / 2;
			objects[kk] = { objects[k].id, 0,{ hp.x, hp.y, hp.z }, {0,0,0,0}, nullptr };
			object_map.insert({ objects[kk].p.x, objects[kk].p.y }, kk);
		}
	}
}

//TrackedFrame::Object * TrackedFrame::operator[](uint32_t id) {
//	uint32_t i = (id >> 8) & (N_OBJECTS-1);
//	if (objects[i].id == id)
//		return objects + i;
//	return nullptr;
//}
//
//const TrackedFrame::Object * TrackedFrame::operator[](uint32_t id) const {
//	uint32_t i = (id >> 8) & (N_OBJECTS - 1);
//	if (objects[i].id == id)
//		return objects + i;
//	return nullptr;
//}

TrackedFrame::Object* TrackedFrame::operator()(const Vec3f& p, const Quaternion& q, TrackedFrame::ObjectType t) {
	return operator()(p, q, TRACKING_RAD, TRACKING_QUAT, t);
}

TrackedFrame::Object* TrackedFrame::operator()(const Vec3f& p, const Quaternion& q, float radius, float angular_dist, TrackedFrame::ObjectType t) {
	Object* match = nullptr;

	float closest_dist = radius * radius;

	object_map.find({ p.x, p.y }, [&](size_t i) {
		Object* candidate = objects + i;

		float dist = D2(candidate->p, p);

		bool close_enough_pos = dist < closest_dist;
		bool close_enough_ang = D2(candidate->q, q) < angular_dist;
		bool match_type = (candidate->type() == t || t == UNKNOWN);

		if (close_enough_pos && close_enough_ang && match_type) {
			closest_dist = dist;
			match = candidate;
		}
	});

	return match;
}

TrackedFrame::Object* TrackedFrame::operator()(const Vec3f &p, const Quaternion &q) {
	return operator()(p, q, TRACKING_RAD, TRACKING_QUAT, UNKNOWN);
}

const TrackedFrame::Object* TrackedFrame::operator()(const Vec3f &p, const Quaternion &q) const {
	return ((TrackedFrame*)this)->operator()(p, q);
}

// Object type in the 4 leftmost bits.
TrackedFrame::ObjectType TrackedFrame::Object::type() const {
	return (TrackedFrame::ObjectType)((id >> 28) & 0xf);
}

uint32_t TrackedFrame::Object::handle() const {
	return id & ((1 << 28) - 1);
}

TrackedFrame::PrivateData::~PrivateData() { }