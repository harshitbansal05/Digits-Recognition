package com.example.harshit.svhmtensorflow;

import org.json.JSONObject;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.http.Query;

public interface BoxDetectionService {

    @POST("detect/")
    @FormUrlEncoded
    Call<JSONObject> getBoxes(
            @Field("score_map") String scoreMap,
            @Field("geo_map") String geoMap,
            @Field("map_width") int mapWidth,
            @Field("map_height") int mapHeight,
            @Field("ratio_h") float ratioH,
            @Field("ratio_w") float ratioW
    );
}
