package com.example.harshit.svhmtensorflow;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.converter.scalars.ScalarsConverterFactory;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;

    private static final int DIGIT_INPUT_SIZE = 64;
    private static final int DIGIT_NUM_CLASSES = 6;

    private static final String DIGIT_INPUT_NAME = "image_placeholder";
    private static final String DIGIT_OUTPUT_NAME = "final_logits";

    private static final String MAP_MODEL_FILE = "file:///android_asset/map_model_graph.pb";
    private static final String DIGIT_MODEL_FILE = "file:///android_asset/svhn_model_graph.pb";

    private Executor executor = Executors.newSingleThreadExecutor();
    private TensorflowImageClassifier classifier;

    private TextView resultTextView;
    private ImageView mImageView;
    private Bitmap bitmap;

    private float ratioH, ratioW;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultTextView = findViewById(R.id.result);
        mImageView = findViewById(R.id.image_view);
        initTensorFlowAndLoadModel();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
           final Uri uri = data.getData();
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                mImageView.setImageBitmap(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void dispatchTakePictureIntent(View v) {
        int permission = ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    100
            );
            return;
        }
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), REQUEST_IMAGE_CAPTURE);
    }

    private void resizeBitmap() {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int resizedWidth = width % 32 == 0 ? width : (width / 32) * 32;
        int resizedHeight = height % 32 == 0 ? height : (height / 32 - 1) * 32;
        bitmap = Bitmap.createScaledBitmap(bitmap, resizedWidth, resizedHeight, false);
        ratioW = (float) resizedWidth / (float) width;
        ratioH = (float) resizedHeight / (float) height;
    }

    private void getMaps(Bitmap bitmap, int width, int height, float[] boxes) {
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        int count = -1;
        float[] rgbValues = new float[width * height * 3];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                int p = pixels[index];
                int R = (p & 0xff0000) >> 16;
                int G = (p & 0xff00) >> 8;
                int B = p & 0xff;
                rgbValues[++count] = R;
                rgbValues[++count] = G;
                rgbValues[++count] = B;
            }
        }
        classifier.getMaps(rgbValues, boxes, width, height);
    }

    private void getBoxes(float[] boxes, final float width, final float height, int sum) {
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("http://box2-dev.us-east-2.elasticbeanstalk.com/")
                .addConverterFactory(ScalarsConverterFactory.create())
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        BoxDetectionService service = retrofit.create(BoxDetectionService.class);
        StringBuilder sm = new StringBuilder();
        for (int i = 0; i < sum; i++) sm.append(Float.toString(boxes[i])).append(",");
        Call<String> callback = service.getBoxes(sm.toString(), height, width, ratioH, ratioW);
        callback.enqueue(new Callback<String>() {
            @Override
            public void onResponse(@NonNull Call<String> call, @NonNull Response<String> response) {
                if (response.body() != null) {
                    String responseString = response.body();
                    String[] boxStringArray = responseString.split(",");
                    if (boxStringArray.length == 1) {
                        Toast.makeText(MainActivity.this, "No box detected", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    getDigits(width, height, boxStringArray);
                } else
                    Toast.makeText(MainActivity.this, "No box detected", Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onFailure(@NonNull Call<String> call, @NonNull Throwable t) {
                t.printStackTrace();
                Toast.makeText(MainActivity.this, t.getMessage(), Toast.LENGTH_LONG).show();
            }
        });
    }

    public void detectNumbers(View v) {
        if (bitmap == null) {
            Toast.makeText(this, "Bitmap is null", Toast.LENGTH_SHORT).show();
            return;
        }

        resizeBitmap();
        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        mutableBitmap.setConfig(Bitmap.Config.ARGB_8888);

        float[] boxes = new float[2700];
        getMaps(mutableBitmap, mutableBitmap.getWidth(), mutableBitmap.getHeight(), boxes);
        int sum = 2700;
        for (int i = 0; i < boxes.length; i+=9) {
            float a = boxes[i];
            int count = 0;
            for (int j = 0; j < 9; j++) {
                if (boxes[i+j] == a) count++;
                else break;
            }
            if (count == 9) {
                sum = i;
                break;
            }
        }
        getBoxes(boxes, mutableBitmap.getWidth(), mutableBitmap.getHeight(), sum);
    }

    private void getDigitsDirect() {
        Bitmap croppedBitmap = Bitmap.createScaledBitmap(bitmap, DIGIT_INPUT_SIZE, DIGIT_INPUT_SIZE, true);
        Bitmap smallMutableBitmap = croppedBitmap.copy(Bitmap.Config.ARGB_8888, true);
        smallMutableBitmap.setConfig(Bitmap.Config.ARGB_8888);

        int width = DIGIT_INPUT_SIZE;
        int height = DIGIT_INPUT_SIZE;
        float[] rgbValues = new float[width * height * 3];
        int[] pix = new int[width * height];
        smallMutableBitmap.getPixels(pix, 0, width, 0, 0, width, height);

        int count = -1;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                int p = pix[index];
                int R = (p & 0xff0000) >> 16;
                int G = (p & 0xff00) >> 8;
                int B = p & 0xff;
                rgbValues[++count] = R;
                rgbValues[++count] = G;
                rgbValues[++count] = B;
            }
        }

        int[] finalOutput = classifier.recognizeImage(rgbValues);
        resultTextView.append("Length: " + Integer.toString(finalOutput[0]) + "\n");
        resultTextView.append(Integer.toString(finalOutput[1]) + " ");
        resultTextView.append(Integer.toString(finalOutput[2]) + " ");
        resultTextView.append(Integer.toString(finalOutput[3]) + " ");
        resultTextView.append(Integer.toString(finalOutput[4]) + " ");
        resultTextView.append(Integer.toString(finalOutput[5]) + "\n");
    }

    private void getDigits(float w, float h, String[] boxes) {
        resultTextView.setText("");
        int numBoxes = boxes.length;
        for (int i = 0; i < numBoxes; i += 8) {
            int x_min = Math.min(Integer.parseInt(boxes[i]), Integer.parseInt(boxes[i + 6]));
            int x_max = Math.max(Integer.parseInt(boxes[i + 2]), Integer.parseInt(boxes[i + 4]));
            int y_min = Math.min(Integer.parseInt(boxes[i + 1]), Integer.parseInt(boxes[i + 3]));
            int y_max = Math.max(Integer.parseInt(boxes[i + 5]), Integer.parseInt(boxes[i + 7]));

            x_min = Math.max(0, x_min);
            x_max = Math.max(0, x_max);
            y_min = Math.max(0, y_min);
            y_max = Math.max(0, y_max);

            x_min = Math.min(x_min, (int) w);
            x_max = Math.min(x_max, (int) w);
            y_min = Math.min(y_min, (int) h);
            y_max = Math.min(y_max, (int) h);

            Bitmap croppedBitmap = Bitmap.createBitmap(bitmap, x_min, y_min, x_max - x_min, y_max - y_min);
            croppedBitmap = Bitmap.createScaledBitmap(croppedBitmap, DIGIT_INPUT_SIZE, DIGIT_INPUT_SIZE, true);
            Bitmap smallMutableBitmap = croppedBitmap.copy(Bitmap.Config.ARGB_8888, true);
            smallMutableBitmap.setConfig(Bitmap.Config.ARGB_8888);

            int width = DIGIT_INPUT_SIZE;
            int height = DIGIT_INPUT_SIZE;
            float[] rgbValues = new float[width * height * 3];
            int[] pix = new int[width * height];
            smallMutableBitmap.getPixels(pix, 0, width, 0, 0, width, height);

            int count = -1;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int index = y * width + x;
                    int p = pix[index];
                    int R = (p & 0xff0000) >> 16;
                    int G = (p & 0xff00) >> 8;
                    int B = p & 0xff;
                    rgbValues[++count] = R;
                    rgbValues[++count] = G;
                    rgbValues[++count] = B;
                }
            }

            int[] finalOutput = classifier.recognizeImage(rgbValues);
            resultTextView.append("Length: " + Integer.toString(finalOutput[0]) + "\n");
            resultTextView.append(Integer.toString(finalOutput[1]) + " ");
            resultTextView.append(Integer.toString(finalOutput[2]) + " ");
            resultTextView.append(Integer.toString(finalOutput[3]) + " ");
            resultTextView.append(Integer.toString(finalOutput[4]) + " ");
            resultTextView.append(Integer.toString(finalOutput[5]) + "\n");
        }
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorflowImageClassifier.create(
                            getAssets(),
                            MAP_MODEL_FILE,
                            DIGIT_MODEL_FILE,
                            DIGIT_INPUT_SIZE,
                            DIGIT_NUM_CLASSES,
                            DIGIT_INPUT_NAME,
                            DIGIT_OUTPUT_NAME);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });
    }
}
