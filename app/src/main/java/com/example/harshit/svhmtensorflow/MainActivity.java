package com.example.harshit.svhmtensorflow;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;

    private static final int BOX_INPUT_SIZE = 448;
    private static final int BOX_NUM_CLASSES = 4;
    private static final int DIGIT_INPUT_SIZE = 64;
    private static final int DIGIT_NUM_CLASSES = 6;

    private static final String BOX_INPUT_NAME = "image";
    private static final String BOX_OUTPUT_NAME = "final_logits";
    private static final String DIGIT_INPUT_NAME = "image_placeholder";
    private static final String DIGIT_OUTPUT_NAME = "final_logits";

    private static final String BOX_MODEL_FILE = "file:///android_asset/box_model_graph.pb";
    private static final String DIGIT_MODEL_FILE = "file:///android_asset/svhm_model_graph.pb";

    private Executor executor = Executors.newSingleThreadExecutor();
    private TensorflowImageClassifier classifier;

    private TextView resultTextView;
    private ImageView mImageView;
    private Bitmap resizedBitmap;

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
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                resizedBitmap = Bitmap.createScaledBitmap(bitmap, BOX_INPUT_SIZE, BOX_INPUT_SIZE, true);
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

    public void detectNumbers(View v) {
        if (resizedBitmap == null) {
            return;
        }
        Bitmap mutableBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888, true);
        mutableBitmap.setConfig(Bitmap.Config.ARGB_8888);

        int boxWidth = BOX_INPUT_SIZE;
        int boxHeight = BOX_INPUT_SIZE;
        float[] boxRgbValues = new float[boxWidth  * boxHeight * 3];
        int[] pixels = new int[boxHeight * boxWidth];
        mutableBitmap.getPixels(pixels, 0, boxWidth, 0, 0, boxWidth, boxHeight);

        int count = -1;
        for (int y = 0; y < boxHeight; y++) {
            for (int x = 0; x < boxWidth; x++) {
                int index = y * boxWidth + x;
                int p = pixels[index];
                int R = (p & 0xff0000) >> 16;
                int G = (p & 0xff00) >> 8;
                int B = p & 0xff;
                boxRgbValues[++count] = R;
                boxRgbValues[++count] = G;
                boxRgbValues[++count] = B;
            }
        }

        float[] output = classifier.recognizeBox(boxRgbValues);
        Bitmap croppedBitmap =
                Bitmap.createBitmap(mutableBitmap, (int) output[0], (int) output[1], (int) output[2], (int) output[3]);
        croppedBitmap = Bitmap.createScaledBitmap(croppedBitmap, DIGIT_INPUT_SIZE, DIGIT_INPUT_SIZE, true);
        Bitmap smallMutableBitmap = croppedBitmap.copy(Bitmap.Config.ARGB_8888, true);
        smallMutableBitmap.setConfig(Bitmap.Config.ARGB_8888);

        int width = DIGIT_INPUT_SIZE;
        int height = DIGIT_INPUT_SIZE;
        float[] rgbValues = new float[width * height * 3];
        int[] pix = new int[width * height];
        smallMutableBitmap.getPixels(pix, 0, width, 0, 0, width, height);

        count = -1;
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

        resultTextView.setText("Length: " + Integer.toString(finalOutput[0]) + "\n");
        resultTextView.append(Integer.toString(finalOutput[1]) + " ");
        resultTextView.append(Integer.toString(finalOutput[2]) + " ");
        resultTextView.append(Integer.toString(finalOutput[3]) + " ");
        resultTextView.append(Integer.toString(finalOutput[4]) + " ");
        resultTextView.append(Integer.toString(finalOutput[5]));
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorflowImageClassifier.create(
                            getAssets(),
                            BOX_MODEL_FILE,
                            DIGIT_MODEL_FILE,
                            BOX_INPUT_SIZE,
                            DIGIT_INPUT_SIZE,
                            BOX_NUM_CLASSES,
                            DIGIT_NUM_CLASSES,
                            BOX_INPUT_NAME,
                            DIGIT_INPUT_NAME,
                            BOX_OUTPUT_NAME,
                            DIGIT_OUTPUT_NAME);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });
    }
}
