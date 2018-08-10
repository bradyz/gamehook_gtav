cbuffer IDBuffer: register(b0) {
	uint base_id;
};

cbuffer velocity_matrix_buffer: register(b1) {
	row_major float4x4 prevViewProj;
	row_major float4x4 prevViewProjInv;

	row_major float4x4 curViewProj;
	row_major float4x4 curViewProjInv;
};

void main(in float4 cur_pos: SV_Position, in float4 prev_pos: PREV_POSITION, out float4 flow_disp: SV_Target5, out uint id_out: SV_Target6) {
	id_out = base_id;

	flow_disp.xyz = prev_pos.xyz / prev_pos.w;
	flow_disp.w = cur_pos.z / cur_pos.w;
}