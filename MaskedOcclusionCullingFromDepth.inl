////////////////////////////////////////////////////////////////////////////////
// Extension: hierarchical-Z fill from dense per-pixel 1/w (clip-space w reciprocal).
// Included inside each SIMD TU namespace after MaskedOcclusionCullingCommon.inl.
// No standard headers here — includes inside this namespace break ADL / nested std.
////////////////////////////////////////////////////////////////////////////////

class MaskedOcclusionCullingFromDepthPrivate : public MaskedOcclusionCullingPrivate
{
public:
	using MaskedOcclusionCullingPrivate::MaskedOcclusionCullingPrivate;

	void ImportPixelDepthBuffer(const float *pixelRcpW, bool flipY) override;
};

void MaskedOcclusionCullingFromDepthPrivate::ImportPixelDepthBuffer(const float *pixelRcpW, bool flipY)
{
	assert(mMaskedHiZBuffer != nullptr && pixelRcpW != nullptr);
	assert(TILE_HEIGHT % SUB_TILE_HEIGHT == 0);
	assert(TILE_WIDTH % SUB_TILE_WIDTH == 0);
	const int nSubtileRows = TILE_HEIGHT / SUB_TILE_HEIGHT;
	(void)nSubtileRows;
	assert(SIMD_LANES == 4 * nSubtileRows);

	auto sample = [&](int x, int y) -> float {
		const int rowInBuf = flipY ? (mHeight - y - 1) : y;
		return pixelRcpW[rowInBuf * mWidth + x];
	};

	for (int ty = 0; ty < mTilesHeight; ty++)
	{
		for (int tx = 0; tx < mTilesWidth; tx++)
		{
			const int tileIdx = ty * mTilesWidth + tx;
			ZTile &tile = mMaskedHiZBuffer[tileIdx];

			for (int lane = 0; lane < SIMD_LANES; lane++)
			{
				const int sty = lane / 4;
				const int stx = lane % 4;
				float minZ = FLT_MAX;
				int validCnt = 0;

				for (int py = 0; py < SUB_TILE_HEIGHT; py++)
				{
					for (int px = 0; px < SUB_TILE_WIDTH; px++)
					{
						const int x = tx * TILE_WIDTH + stx * SUB_TILE_WIDTH + px;
						const int y = ty * TILE_HEIGHT + sty * SUB_TILE_HEIGHT + py;
						if (x >= mWidth || y >= mHeight)
							continue;
						const float z = sample(x, y);
						if (z > 0.f && z == z && z < FLT_MAX)
						{
							if (validCnt == 0 || z < minZ)
								minZ = z;
							++validCnt;
						}
					}
				}

				if (validCnt == 0)
				{
					simd_f32(tile.mZMin[0])[lane] = -1.f;
#if QUICK_MASK != 0
					simd_f32(tile.mZMin[1])[lane] = FLT_MAX;
#else
					simd_f32(tile.mZMin[1])[lane] = 0.f;
#endif
				}
				else
				{
					simd_f32(tile.mZMin[0])[lane] = minZ;
#if QUICK_MASK != 0
					simd_f32(tile.mZMin[1])[lane] = FLT_MAX;
#else
					simd_f32(tile.mZMin[1])[lane] = 0.f;
#endif
				}
				simd_i32(tile.mMask)[lane] = 0;
			}
		}
	}
}
