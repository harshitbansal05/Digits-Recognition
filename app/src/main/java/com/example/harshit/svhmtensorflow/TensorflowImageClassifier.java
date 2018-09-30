package com.example.harshit.svhmtensorflow;

import android.content.res.AssetManager;
import android.support.v4.os.TraceCompat;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;

class TensorflowImageClassifier {

    private int digitInputSize;

    private String[] boxInputNames;
    private String digitInputName;
    private String[] boxOutputNames;
    private String[] digitOutputNames;
    private int[] digitOutputs;

    private TensorFlowInferenceInterface mapInferenceInterface, digitInferenceInterface;

    static TensorflowImageClassifier create(
            AssetManager assetManager,
            String mapModelFilename,
            String digitModelFilename,
            int digitInputSize,
            int digitNumClasses,
            String[] boxInputNames,
            String digitInputName,
            String boxOutputName,
            String digitOutputName)
            throws IOException {
        TensorflowImageClassifier c = new TensorflowImageClassifier();
        c.boxInputNames = boxInputNames;
        c.digitInputName = digitInputName;
        c.boxOutputNames = new String[]{boxOutputName};
        c.digitOutputNames = new String[]{digitOutputName};

        c.mapInferenceInterface = new TensorFlowInferenceInterface(assetManager, mapModelFilename);
        c.digitInferenceInterface = new TensorFlowInferenceInterface(assetManager, digitModelFilename);

        c.digitInputSize = digitInputSize;
        c.digitOutputs = new int[digitNumClasses];
        return c;
    }

    void getMaps(final float[] pixels, float[] boxes, int width, int height) {
        TraceCompat.beginSection("getMaps");

        // Copy the input data into TensorFlow.
        TraceCompat.beginSection("feed");
        mapInferenceInterface.feed(boxInputNames[0], pixels, new long[]{width * height * 3});
        mapInferenceInterface.feed(boxInputNames[1], new int[]{width}, new long[]{1});
        mapInferenceInterface.feed(boxInputNames[2], new int[]{height}, new long[]{1});
        TraceCompat.endSection();

        // Run the inference call.
        TraceCompat.beginSection("run");
        mapInferenceInterface.run(boxOutputNames, false);
        TraceCompat.endSection();

        // Copy the output Tensor back into the output array.
        TraceCompat.beginSection("fetch");
        mapInferenceInterface.fetch(boxOutputNames[0], boxes);
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
