Texture2D<float> prev_disp: register(t1);
Texture2D<float4> flow_disp: register(t0);

SamplerState S : register(s0) {
	Filter = MIN_MAG_MIP_LINEAR;
	AddressU = BORDER;
	AddressV = BORDER;
	BorderColor = float4(0, 0, 0, 0);
};

// Correct the disparity disp_a * (z + disp_b)
cbuffer disparity_correction {
	float disp_a = 1.0;
	float disp_b = 0.0;
};

cbuffer velocity_matrix_buffer {
	row_major float4x4 prevViewProj;
	row_major float4x4 prevViewProjInv;

	row_major float4x4 curViewProj;
	row_major float4x4 curViewProjInv;
};

void main(in float4 p : SV_Position, in float2 t : TEX_COORD, out float2 flow : SV_Target0, out float disparity : SV_Target1, out float occlusion : SV_Target2, out float4 velocity : SV_Target3) {
	uint W, H;

	flow_disp.GetDimensions(W, H);

	int x = t.x * (W - 1);
	int y = t.y * (H - 1);

	float4 f = flow_disp.Load(int3(x, y, 0));

	// Map from ndc coordinates [-1, 1] -> [0, 1].
	// Note: y coordinate is flipped.
	float X_0_1 = (f.x + 1) * 0.5;
	float Y_0_1 = (1 - f.y) * 0.5;

	// Map to screen coordinates.
	float X_0_W = X_0_1 * W;
	float Y_0_H = Y_0_1 * H;

	float D_cur = f.w;
	float D_prev = f.z;

	if (D_prev <= 0.0)
		flow = 0. / 0.; // NaN
	else
		flow = float2(X_0_W - x, Y_0_H - y) - 0.5;

	float4 prev_pos = float4(X_0_1 * 2.0 - 1.0, Y_0_1 * 2.0 - 1.0, D_prev, 1.0);
	float4 cur_pos = float4(t.x * 2.0 - 1.0, t.y * 2.0 - 1.0, D_cur, 1.0);

	float4 cur_prev_pos = mul(mul(cur_pos, curViewProjInv), prevViewProj);
	float4 prev_cur_pos = mul(mul(prev_pos, prevViewProjInv), curViewProj);

	float4 cur_world = mul(cur_pos, curViewProjInv);
	float4 prev_world = mul(prev_pos, prevViewProjInv);

	float4 flow_3d = cur_pos - prev_pos;
	float4 static_flow = cur_pos - cur_prev_pos;

	float4 phil = flow_3d - static_flow;
	float4 mine = cur_pos - prev_cur_pos;

	// velocity = (float4) mul((float1x4) cur_pos, curViewProjInv);

	if (D_prev <= 0.0)
		velocity = 0. / 0.;
	else
		velocity = phil;

	float prev_disparity;

	disparity = disp_a * (D_cur + disp_b);
	prev_disparity = disp_a * (D_prev + disp_b);

	// Temporarily disabled.
	occlusion = (1. / prev_disparity) - (1. / prev_disp.Sample(S, float2(X_0_1, Y_0_1)));
}