package com.example.harshit.svhmtensorflow;

import android.content.res.AssetManager;
import android.support.v4.os.TraceCompat;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;

class TensorflowImageClassifier {

    private int digitInputSize;

    private String digitInputName;
    private String[] digitOutputNames;
    private int[] digitOutputs;

    private TensorFlowInferenceInterface mapInferenceInterface, digitInferenceInterface;

    static TensorflowImageClassifier create(
            AssetManager assetManager,
            String mapModelFilename,
            String digitModelFilename,
            int digitInputSize,
            int digitNumClasses,
            String digitInputName,
            String digitOutputName)
            throws IOException {
        TensorflowImageClassifier c = new TensorflowImageClassifier();
        c.digitInputName = digitInputName;
        c.digitOutputNames = new String[]{digitOutputName};

        c.mapInferenceInterface = new TensorFlowInferenceInterface(assetManager, mapModelFilename);
        c.digitInferenceInterface = new TensorFlowInferenceInterface(assetManager, digitModelFilename);

        c.digitInputSize = digitInputSize;
        c.digitOutputs = new int[digitNumClasses];
        return c;
    }

    void getMaps(final float[] pixels, float[] scoreMap, float[] geometryMap, int width, int height) {
        TraceCompat.beginSection("getMaps");

        // Copy the input data into TensorFlow.
        TraceCompat.beginSection("feed");
        mapInferenceInterface.feed("input_image", pixels, new long[]{width * height * 3});
        mapInferenceInterface.feed("width", new int[]{width}, new long[]{1});
        mapInferenceInterface.feed("height", new int[]{height}, new long[]{1});
        TraceCompat.endSection();

        // Run the inference call.
        TraceCompat.beginSection("run");
        mapInferenceInterface.run(new String[]{"F_score", "F_geometry"}, false);
        TraceCompat.endSection();

        // Copy the output Tensor back into the output array.
        TraceCompat.beginSection("fetch");
        mapInferenceInterface.fetch("F_score", scoreMap);
        mapInferenceInterface.fetch("F_geometry", geometryMap);
        TraceCompat.endSection();
    }

    int[] recognizeImage(final float[] pixels) {
        TraceCompat.beginSection("recognizeImage");

        // Copy the input data into TensorFlow.
        TraceCompat.beginSection("feed");
        digitInferenceInterface.feed(digitInputName, pixels, new long[]{digitInputSize * digitInputSize * 3});
        TraceCompat.endSection();

        // Run the inference call.
        TraceCompat.beginSection("run");
        digitInferenceInterface.run(digitOutputNames, false);
        TraceCompat.endSection();

        // Copy the output Tensor back into the output array.
        TraceCompat.beginSection("fetch");
        digitInferenceInterface.fetch(digitOutputNames[0], digitOutputs);
        TraceCompat.endSection();
        return digitOutputs;
    }
}
