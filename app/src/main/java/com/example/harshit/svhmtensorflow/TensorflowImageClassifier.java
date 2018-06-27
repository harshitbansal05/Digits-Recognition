package com.example.harshit.svhmtensorflow;

import android.content.res.AssetManager;
import android.support.v4.os.TraceCompat;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;

class TensorflowImageClassifier {

    private int boxInputSize, digitInputSize;

    private String boxInputName, digitInputName;
    private String[] boxOutputNames, digitOutputNames;
    private float[] boxOutputs;
    private int[] digitOutputs;

    private TensorFlowInferenceInterface boxInferenceInterface, digitInferenceInterface;

    static TensorflowImageClassifier create(
            AssetManager assetManager,
            String boxModelFilename,
            String digitModelFilename,
            int boxInputSize,
            int digitInputSize,
            int boxNumClasses,
            int digitNumClasses,
            String boxInputName,
            String digitInputName,
            String boxOutputName,
            String digitOutputName)
            throws IOException {
        TensorflowImageClassifier c = new TensorflowImageClassifier();
        c.boxInputName = boxInputName;
        c.digitInputName = digitInputName;
        c.boxOutputNames = new String[]{boxOutputName};
        c.digitOutputNames = new String[]{digitOutputName};

        c.boxInferenceInterface = new TensorFlowInferenceInterface(assetManager, boxModelFilename);
        c.digitInferenceInterface = new TensorFlowInferenceInterface(assetManager, digitModelFilename);

        c.boxInputSize = boxInputSize;
        c.digitInputSize = digitInputSize;

        c.boxOutputs = new float[boxNumClasses];
        c.digitOutputs = new int[digitNumClasses];
        return c;
    }

    float[] recognizeBox(final float[] pixels) {
        TraceCompat.beginSection("recognizeImage");

        // Copy the input data into TensorFlow.
        TraceCompat.beginSection("feed");
        boxInferenceInterface.feed(boxInputName, pixels, new long[]{boxInputSize * boxInputSize * 3});
        TraceCompat.endSection();

        // Run the inference call.
        TraceCompat.beginSection("run");
        boxInferenceInterface.run(boxOutputNames, false);
        TraceCompat.endSection();

        // Copy the output Tensor back into the output array.
        TraceCompat.beginSection("fetch");
        boxInferenceInterface.fetch(boxOutputNames[0], boxOutputs);
        TraceCompat.endSection();
        return boxOutputs;
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
