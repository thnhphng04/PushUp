/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.max
import kotlin.math.min
import com.google.mediapipe.examples.poselandmarker.SquatAnalyzer
import android.graphics.RectF
import android.graphics.DashPathEffect

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: PoseLandmarkerResult? = null
    private var pointPaint = Paint()
    private var linePaint = Paint()
    // Thêm feedback
    private var feedback: SquatAnalyzer.Feedback? = null

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    init {
        initPaints()
    }

    fun clear() {
        results = null
        pointPaint.reset()
        linePaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        linePaint.color =
            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        results?.let { poseLandmarkerResult ->
            val landmarkList = poseLandmarkerResult.landmarks().firstOrNull()
            if (landmarkList != null) {
                fun lm(idx: Int): Pair<Float, Float> {
                    val pt = landmarkList.getOrNull(idx)
                    return if (pt != null) Pair(pt.x() * imageWidth * scaleFactor, pt.y() * imageHeight * scaleFactor) else Pair(0f, 0f)
                }
                // Lấy các điểm trái, phải, nose
                val leftShoulder = lm(11)
                val leftElbow = lm(13)
                val leftWrist = lm(15)
                val leftHip = lm(23)
                val leftKnee = lm(25)
                val leftAnkle = lm(27)
                val leftFoot = lm(31)
                val leftEar = lm(7)
                val rightShoulder = lm(12)
                val rightElbow = lm(14)
                val rightWrist = lm(16)
                val rightHip = lm(24)
                val rightKnee = lm(26)
                val rightAnkle = lm(28)
                val rightFoot = lm(32)
                val rightEar = lm(8)
                val nose = lm(0)

                if (feedback?.cameraWarning == true) {
                    // Vẽ landmark mũi, vai trái, vai phải và các đường nối
                    val nosePaint = Paint(pointPaint); nosePaint.color = Color.WHITE
                    val leftPaint = Paint(pointPaint); leftPaint.color = Color.YELLOW
                    val rightPaint = Paint(pointPaint); rightPaint.color = Color.CYAN
                    canvas.drawCircle(nose.first, nose.second, 18f, nosePaint)
                    canvas.drawCircle(leftShoulder.first, leftShoulder.second, 18f, leftPaint)
                    canvas.drawCircle(rightShoulder.first, rightShoulder.second, 18f, rightPaint)
                    // Vẽ đường nối giữa 3 điểm
                    val connectPaint = Paint(linePaint); connectPaint.color = Color.MAGENTA; connectPaint.strokeWidth = 8f
                    canvas.drawLine(leftShoulder.first, leftShoulder.second, rightShoulder.first, rightShoulder.second, connectPaint)
                    canvas.drawLine(nose.first, nose.second, leftShoulder.first, leftShoulder.second, connectPaint)
                    canvas.drawLine(nose.first, nose.second, rightShoulder.first, rightShoulder.second, connectPaint)
                } else {
                    // VẼ ĐẦY ĐỦ ĐỘNG TÁC BÊN TRỤ
                    val distL = kotlin.math.abs(leftFoot.second - leftShoulder.second)
                    val distR = kotlin.math.abs(rightFoot.second - rightShoulder.second)
                    val isLeft = distL > distR
                    val points = if (isLeft) {
                        listOf(leftEar, leftShoulder, leftElbow, leftWrist, leftHip, leftKnee, leftAnkle, leftFoot)
                    } else {
                        listOf(rightEar, rightShoulder, rightElbow, rightWrist, rightHip, rightKnee, rightAnkle, rightFoot)
                    }
                    val ear = points[0]
                    val shldr = points[1]
                    val elbow = points[2]
                    val wrist = points[3]
                    val hip = points[4]
                    val knee = points[5]
                    val ankle = points[6]
                    val foot = points[7]

                    // Vẽ các đường nối bên trụ
                    val jointPaint = Paint(linePaint)
                    jointPaint.strokeWidth = 8f
                    jointPaint.color = Color.parseColor("#66ccff")
                    canvas.drawLine(ear.first, ear.second, shldr.first, shldr.second, jointPaint)
                    canvas.drawLine(shldr.first, shldr.second, hip.first, hip.second, jointPaint)
                    canvas.drawLine(hip.first, hip.second, knee.first, knee.second, jointPaint)
                    canvas.drawLine(knee.first, knee.second, ankle.first, ankle.second, jointPaint)

                    // Vẽ đường nối ear-elbow và elbow-hip
                    val earElbowPaint = Paint(linePaint)
                    earElbowPaint.strokeWidth = 6f
                    earElbowPaint.color = Color.parseColor("#FF6B6B") // Màu đỏ cam
                    canvas.drawLine(ear.first, ear.second, elbow.first, elbow.second, earElbowPaint)
                    
                    val elbowHipPaint = Paint(linePaint)
                    elbowHipPaint.strokeWidth = 6f
                    elbowHipPaint.color = Color.parseColor("#FF6B6B") // Màu xanh lá
                    canvas.drawLine(elbow.first, elbow.second, hip.first, hip.second, elbowHipPaint)

                    // Vẽ các điểm landmark bên trụ
                    val mainPaint = Paint(pointPaint)
                    mainPaint.color = if (isLeft) Color.YELLOW else Color.CYAN
                    listOf(shldr, elbow, wrist, hip, knee, ankle, foot, ear).forEach {
                        canvas.drawCircle(it.first, it.second, 14f, mainPaint)
                    }


                    // Vẽ số liệu các góc tại vị trí tương ứng
                    val anglePaint = Paint().apply {
                        color = Color.GREEN
                        textSize = 48f
                        style = Paint.Style.FILL
                        setShadowLayer(8f, 0f, 0f, Color.BLACK)
                    }

                    canvas.drawText("${feedback?.shldrAngle ?: 0}", shldr.first + 10, shldr.second, anglePaint)
                    canvas.drawText("${feedback?.hipAngle ?: 0}", hip.first + 10, hip.second, anglePaint)
                    canvas.drawText("${feedback?.kneeAngle ?: 0}", knee.first + 15, knee.second + 10, anglePaint)

                    // Vẽ số liệu góc ear-elbow-hip tại elbow
                    val earElbowHipAnglePaint = Paint().apply {
                        color = Color.parseColor("#FF6B6B") // Màu đỏ cam giống cung tròn
                        textSize = 42f
                        style = Paint.Style.FILL
                        setShadowLayer(8f, 0f, 0f, Color.BLACK)
                    }
                    canvas.drawText("${feedback?.earElbowHipAngle ?: 0}", elbow.first + 10, elbow.second + 5, earElbowHipAnglePaint)

                    // Hiển thị các cảnh báo động tác nếu có (ở phía dưới)
                    val feedbacks = feedback?.feedbackList ?: emptyList()
                    if (feedbacks.isNotEmpty()) {
                        val fbPaint = Paint().apply {
                            color = Color.rgb(255, 140, 0)
                            textSize = 70f
                            style = Paint.Style.FILL
                            setShadowLayer(10f, 0f, 0f, Color.BLACK)
                        }
                        feedbacks.forEachIndexed { i, msg ->
                            canvas.drawText(msg, 40f, height - 220f - (feedbacks.size-1-i)*80f, fbPaint)
                        }
                    }
                }

                // Hiển thị lại bộ đếm squat đúng/sai ở phía dưới cùng (luôn luôn hiển thị)
                val countPaint = Paint().apply {
                    color = Color.WHITE
                    textSize = 54f
                    style = Paint.Style.FILL
                    setShadowLayer(10f, 0f, 0f, Color.BLACK)
                }
                val correctBg = Paint().apply {
                    color = Color.rgb(18, 185, 0)
                    style = Paint.Style.FILL
                }
                val incorrectBg = Paint().apply {
                    color = Color.rgb(221, 0, 0)
                    style = Paint.Style.FILL
                }
                val correctText = "PushUp đúng: ${feedback?.correctCount ?: 0}"
                val incorrectText = "PushUp sai: ${feedback?.incorrectCount ?: 0}"
                val padding = 30f
                val bottomY = height - 200f
                val spacing = 80f
                val correctWidth = countPaint.measureText(correctText) + 2*padding
                val incorrectWidth = countPaint.measureText(incorrectText) + 2*padding
                // Nền xanh cho squat đúng
                canvas.drawRoundRect(width-correctWidth-40f, bottomY-spacing-40f, width-40f, bottomY-spacing+30f, 30f, 30f, correctBg)
                // Nền đỏ cho squat sai
                canvas.drawRoundRect(width-incorrectWidth-40f, bottomY-40f, width-40f, bottomY+30f, 30f, 30f, incorrectBg)
                // Text
                canvas.drawText(correctText, width-correctWidth-padding, bottomY-spacing, countPaint)
                canvas.drawText(incorrectText, width-incorrectWidth-padding, bottomY, countPaint)
            }
        }
        // XÓA đoạn vẽ các thông tin squat đúng/sai, hip/knee/ankle angle ở cuối file
    }

    fun setResults(
        poseLandmarkerResults: PoseLandmarkerResult?,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE,
        feedback: SquatAnalyzer.Feedback? = null
    ) {
        results = poseLandmarkerResults
        this.feedback = feedback
        this.imageHeight = imageHeight
        this.imageWidth = imageWidth
        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 12F
    }
}