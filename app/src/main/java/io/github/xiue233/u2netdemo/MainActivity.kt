package io.github.xiue233.u2netdemo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.drawable.Drawable
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.LinearLayout
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.MemoryFormat
import org.pytorch.Module
import org.pytorch.PyTorchAndroid
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private lateinit var root: LinearLayout
    private lateinit var originImg: ImageView

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        root = findViewById(R.id.root)
        originImg = findViewById(R.id.img_origin)
        originImg.setImageDrawable(
            Drawable.createFromStream(
                assets.open("cat.jpg"),
                "cat"
            )
        )
        thread {
            val module = Module.load(assetFilePath(this, "u2netp_small_live_test.ptl"))
            val bitmap = BitmapFactory.decodeStream(
                assets.open("cat.jpg")
            )
            //You can see the code of TensorImageUtils.bitmapToFloatBuffer() to have a better
            //understanding of how the content of a tensor is written.
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                MemoryFormat.CHANNELS_LAST
            )
            //the output of the model is a tuple in which there are several tensors.
            val outputs = module.forward(IValue.from(inputTensor)).toTuple()
            for (value in outputs.iterator()) {
                val outBitmap = transformTensors2Bitmap(value.toTensor())
                runOnUiThread {
                    root.addView(ImageView(this).apply {
                        setImageBitmap(
                            outBitmap
                        )
                    })
                }
            }
        }
    }

    /**
     * Transform a tensor into a bitmap.
     *
     * If you have a look at the returned tensors, you will find the shape of it is [1,1,h,w].
     */
    @RequiresApi(Build.VERSION_CODES.O)
    private fun transformTensors2Bitmap(output: Tensor): Bitmap {
        val height = output.shape()[2].toInt()
        val width = output.shape()[3].toInt()
        val outputArr = output.dataAsFloatArray
        for (i in outputArr.indices) {
            outputArr[i] = Math.min(Math.max(outputArr[i], 0f), 255f)
        }
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565)
        var loc = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                bitmap.setPixel(
                    x, y, Color.rgb(
                        outputArr[loc],
                        outputArr[loc],
                        outputArr[loc]
                    )
                )
                loc += 1
            }
        }
        return bitmap
    }

    private fun assetFilePath(context: Context, assetName: String): String? {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        try {
            context.assets.open(assetName).use { `is` ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * 1024)
                    var read = 0
                    while (`is`.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                    }
                    os.flush()
                }
                return file.absolutePath
            }
        } catch (e: IOException) {
            Log.e("MainActivity", "Error process asset $assetName to file path")
        }
        return null
    }
}