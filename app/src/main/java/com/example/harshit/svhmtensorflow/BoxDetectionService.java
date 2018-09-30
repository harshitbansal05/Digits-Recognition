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
    Call<String> getBoxes(
            @Field("boxes") String boxes,
            @Field("h") float height,
            @Field("w") float width,
            @Field("ratio_h") float ratioH,
            @Field("ratio_w") float ratioW
    );
}
