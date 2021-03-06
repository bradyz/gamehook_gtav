cbuffer IDBuffer: register(b0) {
	uint base_id;
	uint tex_id;
};

void main(in float4 cur_pos: SV_Position, in float4 prev_pos: PREV_POSITION, out uint texture_out: SV_Target5, out float4 flow_disp: SV_Target6, out uint id_out: SV_Target7) {
	id_out = base_id;
    texture_out = tex_id;

	flow_disp.xyz = prev_pos.xyz / prev_pos.w;
	flow_disp.w = cur_pos.z / cur_pos.w;
}