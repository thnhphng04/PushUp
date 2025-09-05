package com.google.mediapipe.examples.poselandmarker

import kotlin.math.*

class SquatAnalyzer(
    private val thresholds: Thresholds = Thresholds.defaultBeginner()
) {
    enum class SquatState { NONE, S1, S2, S3 }

    data class Feedback(
        val correctCount: Int,
        val incorrectCount: Int,
        val message: String,
        val shldrAngle: Int,
        val hipAngle: Int,
        val kneeAngle: Int,
        val earElbowHipAngle: Int = 0,
        val cameraWarning: Boolean = false,
        val offsetAngle: Int = 0,
        val lowerHips: Boolean = false,
        val feedbackList: List<String> = emptyList()
    )

    // State tracking variables
    private var stateSeq = mutableListOf<String>()
    private var correctCount = 0
    private var incorrectCount = 0
    private var improperSquat = false
    private var prevState: String? = null
    private var currState: String? = null
    private var displayText = BooleanArray(3) { false }
    private var countFrames = IntArray(3) { 0 }
    private var lowerHips = false
    private var inactiveTime = 0.0
    private var inactiveTimeFront = 0.0
    private var startInactiveTime = System.nanoTime() / 1e9
    private var startInactiveTimeFront = System.nanoTime() / 1e9
    private var incorrectPosture = false
    private var cameraWarning = false
    private var offsetAngle = 0
    private var feedbackList = mutableListOf<String>()

    fun analyze(landmarks: List<Pair<Float, Float>>, rightLandmarks: List<Pair<Float, Float>>? = null, nose: Pair<Float, Float>? = null): Feedback {
        // Lấy các điểm cần thiết (theo MediaPipe Pose)
        val leftShoulder = getLandmark(landmarks, 11)
        val leftHip = getLandmark(landmarks, 23)
        val leftKnee = getLandmark(landmarks, 25)
        val leftAnkle = getLandmark(landmarks, 27)
        val leftFoot = getLandmark(landmarks, 31)
        val leftEar = getLandmark(landmarks, 7)
        val rightShoulder = rightLandmarks?.let { getLandmark(it, 12) } ?: Pair(0f, 0f)
        val rightHip = rightLandmarks?.let { getLandmark(it, 24) } ?: Pair(0f, 0f)
        val rightKnee = rightLandmarks?.let { getLandmark(it, 26) } ?: Pair(0f, 0f)
        val rightAnkle = rightLandmarks?.let { getLandmark(it, 28) } ?: Pair(0f, 0f)
        val rightFoot = rightLandmarks?.let { getLandmark(it, 32) } ?: Pair(0f, 0f)
        val rightEar = rightLandmarks?.let { getLandmark(it, 8) } ?: Pair(0f, 0f)
        val noseCoord = nose ?: Pair(0f, 0f)

        // Tính offset angle để phát hiện lệch camera
        offsetAngle = findAngle(leftShoulder, noseCoord, rightShoulder)
        cameraWarning = offsetAngle > thresholds.offsetThresh
        feedbackList.clear()

        val now = System.nanoTime() / 1e9
        if (cameraWarning) {
            // Đếm thời gian lệch camera
            inactiveTimeFront += now - startInactiveTimeFront
            startInactiveTimeFront = now
            if (inactiveTimeFront >= thresholds.inactiveThresh) {
                correctCount = 0
                incorrectCount = 0
                inactiveTimeFront = 0.0
            }
            // Feedback cảnh báo camera
            feedbackList.add("CAMERA NOT ALIGNED PROPERLY!!!")
            feedbackList.add("OFFSET ANGLE: $offsetAngle")
            prevState = null
            currState = null
            startInactiveTime = now
            inactiveTime = 0.0
        } else {
            inactiveTimeFront = 0.0
            startInactiveTimeFront = now
            // Chọn bên chân trụ (dựa vào khoảng cách vai-bàn chân)
            val distL = Math.abs(leftFoot.second - leftShoulder.second)
            val distR = Math.abs(rightFoot.second - rightShoulder.second)
            val points = if (distL > distR) {
                listOf(
                    leftEar,
                    leftShoulder,
                    getLandmark(landmarks, 13),
                    getLandmark(landmarks, 15),
                    leftHip,
                    leftKnee,
                    leftAnkle,
                    leftFoot
                )
            } else {
                listOf(
                    rightEar,
                    rightShoulder,
                    getLandmark(rightLandmarks ?: emptyList(), 14),
                    getLandmark(rightLandmarks ?: emptyList(), 16),
                    rightHip,
                    rightKnee,
                    rightAnkle,
                    rightFoot
                )
            }
            val ear = points[0]
            val shldr = points[1]
            val elbow = points[2]
            val wrist = points[3]
            val hip = points[4]
            val knee = points[5]
            val ankle = points[6]
            val foot = points[7]
            // Tính các góc mới
            val elbowAngle = findAngle(shldr, elbow, wrist)
            val shldrAngle = findAngle(ear, shldr, hip)
            val hipAngle = findAngle(shldr, hip, knee)
            val kneeAngle = findAngle(hip, knee, ankle)

            val earElbowHipAngle = findAngle(ear, elbow, hip) // Góc ear-elbow-hip (đỉnh là elbow)
            // State machine
            currState = getState(elbowAngle, earElbowHipAngle)
            updateStateSequence(currState)
            // Đếm squat đúng/sai
            var message = ""
            if (currState == "s1") {
                if (stateSeq.size == 3 && !incorrectPosture) {
                    correctCount++
                    message = "CORRECT"
                } else if (stateSeq.contains("s2") && stateSeq.size == 1) {
                    incorrectCount++
                    message = "INCORRECT"
                } else if (incorrectPosture) {
                    incorrectCount++
                    message = "INCORRECT"
                }
                stateSeq.clear()
                incorrectPosture = false
            } else { //không còn ở state 1
                // Feedback động tác
                if (shldrAngle < thresholds.shldrMin) { displayText[0] = true; incorrectPosture = true; feedbackList.add("BENT NECK") }
                if (hipAngle < thresholds.hipMin) { displayText[1] = true; incorrectPosture = true; feedbackList.add("BENT HIP") }
                if (kneeAngle < thresholds.kneeMin) { displayText[2] = true; incorrectPosture = true; feedbackList.add("BENT KNEE") }

                if (earElbowHipAngle in thresholds.earElbowHipTrans && stateSeq.count { it == "s2" } == 1) { lowerHips = true; feedbackList.add("Continue lowering") }
            }
            // Inactivity logic
            if (currState == prevState) {
                inactiveTime += now - startInactiveTime
                startInactiveTime = now
                if (inactiveTime >= thresholds.inactiveThresh) {
                    correctCount = 0
                    incorrectCount = 0
                }
            } else {
                startInactiveTime = now
                inactiveTime = 0.0
            }
            if (stateSeq.contains("s3") || currState == "s1") lowerHips = false
            prevState = currState
            // Reset feedback nếu quá lâu
            for (i in displayText.indices) {
                if (countFrames[i] > thresholds.cntFrameThresh) {
                    displayText[i] = false
                    countFrames[i] = 0
                }
                if (displayText[i]) countFrames[i]++
            }
            // Trả về feedback chi tiết
            return Feedback(
                correctCount = correctCount,
                incorrectCount = incorrectCount,
                message = message,
                shldrAngle = shldrAngle,
                hipAngle = hipAngle,
                kneeAngle = kneeAngle,
                earElbowHipAngle = earElbowHipAngle,
                cameraWarning = cameraWarning,
                offsetAngle = offsetAngle,
                lowerHips = lowerHips,
                feedbackList = feedbackList.toList()
            )
        }
        // Nếu lệch camera, trả về feedback cảnh báo
        return Feedback(
            correctCount = correctCount,
            incorrectCount = incorrectCount,
            message = "",
            shldrAngle = 0,
            hipAngle = 0,
            kneeAngle = 0,
            earElbowHipAngle = 0,
            cameraWarning = cameraWarning,
            offsetAngle = offsetAngle,
            lowerHips = false,
            feedbackList = feedbackList.toList()
        )
    }

    private fun getLandmark(landmarks: List<Pair<Float, Float>>, idx: Int): Pair<Float, Float> {
        // Nếu truyền vào chỉ có 8 điểm, map lại index
        val mapIdx = when(idx) {
            7, 8 -> 0  // ear
            11, 12 -> 1
            13, 14 -> 2
            15, 16 -> 3
            23, 24 -> 4
            25, 26 -> 5
            27, 28 -> 6
            31, 32 -> 7
            0 -> 0 // nose nếu có
            else -> 0
        }
        return landmarks.getOrNull(mapIdx) ?: Pair(0f, 0f)
    }

    private fun findAngle(p1: Pair<Float, Float>, p2: Pair<Float, Float>, p3: Pair<Float, Float>): Int {
        val a = floatArrayOf(p1.first - p2.first, p1.second - p2.second)
        val b = floatArrayOf(p3.first - p2.first, p3.second - p2.second)
        val dot = a[0]*b[0] + a[1]*b[1]
        val normA = kotlin.math.sqrt(a[0]*a[0] + a[1]*a[1])
        val normB = kotlin.math.sqrt(b[0]*b[0] + b[1]*b[1])
        val cosTheta = (dot / (normA * normB)).coerceIn(-1f, 1f)
        val theta = kotlin.math.acos(cosTheta)
        return Math.toDegrees(theta.toDouble()).toInt()
    }

    private fun angleWithVertical(from: Pair<Float, Float>, to: Pair<Float, Float>): Int {
        val v1 = floatArrayOf(0f, -1f) // vector thẳng đứng hướng lên
        val v2 = floatArrayOf(to.first - from.first, to.second - from.second)
        val dot = v1[0]*v2[0] + v1[1]*v2[1]
        val norm1 = kotlin.math.sqrt(v1[0]*v1[0] + v1[1]*v1[1])
        val norm2 = kotlin.math.sqrt(v2[0]*v2[0] + v2[1]*v2[1])
        val cosTheta = (dot / (norm1 * norm2)).coerceIn(-1f, 1f)
        val theta = kotlin.math.acos(cosTheta)
        return Math.toDegrees(theta.toDouble()).toInt()
    }

    private fun getState(elbowAngle: Int, earElbowHipAngle: Int): String? {
        return when {
            elbowAngle > thresholds.elbowNormal && earElbowHipAngle < thresholds.earElbowHipNormal -> "s1"
            earElbowHipAngle in thresholds.earElbowHipTrans -> "s2"
            earElbowHipAngle in thresholds.earElbowHipPass -> "s3"
            else -> null
        }
    }

    private fun updateStateSequence(state: String?) {
        if (state == null) return
        if (state == "s2") {
            if ((!stateSeq.contains("s3") && stateSeq.count { it == "s2" } == 0) || //thêm s2 khi đang xống
                (stateSeq.contains("s3") && stateSeq.count { it == "s2" } == 1)) {  //thêm s2 khi đang lên
                stateSeq.add(state)
            }
        } else if (state == "s3") {
            if (!stateSeq.contains(state) && stateSeq.contains("s2")) {
                stateSeq.add(state)
            }
        }
    }

    data class Thresholds(
        val elbowNormal: Int,
        val earElbowHipNormal: Int,
        val earElbowHipTrans: IntRange,
        val earElbowHipPass: IntRange,
        val shldrMin: Int,
        val hipMin: Int,
        val kneeMin: Int,
        val offsetThresh: Int = 40,
        val inactiveThresh: Double = 15.0,
        val cntFrameThresh: Int = 50
    ) {
        companion object {
            fun defaultBeginner() = Thresholds(
                elbowNormal = 150,
                earElbowHipNormal = 120,
                earElbowHipTrans = 125..150,
                earElbowHipPass = 155..180,
                shldrMin = 135,
                hipMin = 160,
                kneeMin = 150,
                offsetThresh = 35,
                inactiveThresh = 15.0,
                cntFrameThresh = 50
            )
            fun defaultPro() = Thresholds(
                elbowNormal = 150,
                earElbowHipNormal = 120,
                earElbowHipTrans = 125..150,
                earElbowHipPass = 155..180,
                shldrMin = 150,
                hipMin = 160,
                kneeMin = 160,
                offsetThresh = 35,
                inactiveThresh = 15.0,
                cntFrameThresh = 50
            )
        }
    }
} 