import ArgumentParser
import FluidAudio
import Foundation

@main
struct FluidTranscribeTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "fluid-transcribe",
        abstract: "Transcribe audio with speaker diarization using FluidAudio (Parakeet TDT + pyannote community-1)"
    )

    @Argument(help: "Path to the audio file")
    var audioPath: String

    @Option(name: .long, help: "Number of speakers (auto-detect if omitted)")
    var numSpeakers: Int?

    @Option(name: .shortAndLong, help: "Output file path (default: alongside input with .transcript.txt)")
    var output: String?

    @Flag(name: .long, help: "Verbose output")
    var verbose: Bool = false

    func run() async throws {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Absolute path: the default output path is derived from this, and
        // deletingLastPathComponent on a bare filename returns "" (i.e. "/").
        let resolvedPath = URL(fileURLWithPath: (audioPath as NSString).expandingTildeInPath)
            .standardizedFileURL.path
        guard FileManager.default.fileExists(atPath: resolvedPath) else {
            FileHandle.standardError.write(Data("Error: File not found: \(resolvedPath)\n".utf8))
            throw ExitCode.failure
        }

        if let n = numSpeakers, n < 1 {
            FileHandle.standardError.write(Data("Error: --num-speakers must be at least 1\n".utf8))
            throw ExitCode.validationFailure
        }

        // Resolve and check the output path before any inference runs.
        let rawOutput = output.map { ($0 as NSString).expandingTildeInPath } ?? {
            let dir = (resolvedPath as NSString).deletingLastPathComponent
            let base = ((resolvedPath as NSString).lastPathComponent as NSString).deletingPathExtension
            return "\(dir)/\(base).transcript.txt"
        }()
        // Absolutize so a bare "-o notes.txt" resolves against cwd, not "".
        let outputPath = URL(fileURLWithPath: rawOutput).standardizedFileURL.path
        try validateOutputPath(outputPath)

        let fileName = (resolvedPath as NSString).lastPathComponent
        print("Transcribing: \(fileName)")
        print("Model: FluidInference/parakeet-tdt-0.6b-v3-coreml + pyannote community-1")
        print("")

        // Load audio once, shared by both pipelines. This holds the whole file as
        // Float32 (64 KB per second) instead of using the libraries' streaming
        // URL overloads; fine for hours of audio, not for days.
        print("Loading audio...")
        let loadStart = CFAbsoluteTimeGetCurrent()
        let resolvedURL = URL(fileURLWithPath: resolvedPath)
        let audioSamples = try AudioConverter().resampleAudioFile(resolvedURL)
        guard !audioSamples.isEmpty else {
            FileHandle.standardError.write(Data("Error: no audio samples decoded from \(fileName)\n".utf8))
            throw ExitCode.failure
        }
        let audioDuration = Float(audioSamples.count) / 16000.0
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        print(String(format: "Audio loaded: %.1fs duration (%.1fs to load)", audioDuration, loadTime))
        print("")

        // Run transcription and diarization in parallel
        print("Starting parallel transcription + diarization...")
        print("")

        let numSpeakersLocal = numSpeakers
        let verboseLocal = verbose

        let (asrResult, diarizationResult): (ASRResult, DiarizationResult) = try await withThrowingTaskGroup(
            of: PipelineResult.self
        ) { group in
            // Transcription task
            group.addTask {
                let tStart = CFAbsoluteTimeGetCurrent()
                print("[Transcription] Loading Parakeet TDT v3 models (first run downloads ~600 MB)...")

                let models = try await AsrModels.downloadAndLoad(version: .v3)
                let asrManager = AsrManager(config: .default)
                try await asrManager.loadModels(models)

                if verboseLocal { print("[Transcription] Models loaded, transcribing...") }

                var decoderState = TdtDecoderState.make(decoderLayers: await asrManager.decoderLayerCount)
                let result = try await asrManager.transcribe(audioSamples, decoderState: &decoderState)

                let tTime = CFAbsoluteTimeGetCurrent() - tStart
                print(String(
                    format: "[Transcription] Complete in %.1fs (%.1fx realtime)",
                    tTime, Double(audioDuration) / tTime))

                return .transcription(result)
            }

            // Diarization task
            group.addTask {
                let dStart = CFAbsoluteTimeGetCurrent()
                print("[Diarization] Loading pyannote community-1 models (first run downloads)...")

                var config = OfflineDiarizerConfig()
                if let n = numSpeakersLocal {
                    config = config.withSpeakers(exactly: n)
                }

                let diarizer = OfflineDiarizerManager(config: config)
                try await diarizer.prepareModels()

                if verboseLocal { print("[Diarization] Models loaded, diarizing...") }

                let result = try await diarizer.process(audio: audioSamples)

                let dTime = CFAbsoluteTimeGetCurrent() - dStart
                let speakerCount = Set(result.segments.map { $0.speakerId }).count
                print(String(
                    format: "[Diarization] Complete: %d speakers in %.1fs (%.1fx realtime)",
                    speakerCount, dTime, Double(audioDuration) / dTime))

                return .diarization(result)
            }

            var transcription: ASRResult?
            var diarization: DiarizationResult?

            for try await result in group {
                switch result {
                case .transcription(let t): transcription = t
                case .diarization(let d): diarization = d
                }
            }

            guard let t = transcription, let d = diarization else {
                throw TranscribeError.pipelineFailed
            }
            return (t, d)
        }

        // Align transcription with speaker labels
        print("")
        print("Aligning transcription with speaker labels...")

        let wordTimings = buildWordTimings(from: asrResult.tokenTimings ?? [])

        // Stable speaker numbering by first appearance in the diarization timeline
        let orderedSegments = diarizationResult.segments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
        var speakerNumberById: [String: Int] = [:]
        for seg in orderedSegments where speakerNumberById[seg.speakerId] == nil {
            speakerNumberById[seg.speakerId] = speakerNumberById.count + 1
        }

        func speakerLabel(forTime midpoint: Float) -> String {
            // Exact containment first
            if let hit = orderedSegments.first(where: {
                midpoint >= $0.startTimeSeconds && midpoint < $0.endTimeSeconds
            }) {
                return "Speaker \(speakerNumberById[hit.speakerId] ?? 0)"
            }
            // Fallback: nearest segment by max overlap / minimum distance
            guard !orderedSegments.isEmpty else { return "Unknown" }
            let nearest = orderedSegments.min(by: { seg1, seg2 in
                distance(midpoint, seg1) < distance(midpoint, seg2)
            })!
            return "Speaker \(speakerNumberById[nearest.speakerId] ?? 0)"
        }

        func distance(_ point: Float, _ seg: TimedSpeakerSegment) -> Float {
            if point < seg.startTimeSeconds { return seg.startTimeSeconds - point }
            if point > seg.endTimeSeconds { return point - seg.endTimeSeconds }
            return 0
        }

        // Flatten into (speaker, start, end, word) then merge consecutive same-speaker words
        var flat: [(speaker: String, start: Float, end: Float, text: String)] = []
        for w in wordTimings {
            let midpoint = Float((w.startTime + w.endTime) / 2.0)
            let speaker = speakerLabel(forTime: midpoint)
            flat.append((speaker, Float(w.startTime), Float(w.endTime), w.word))
        }

        let merged = mergeConsecutive(flat)

        // Format output lines
        var lines: [String] = []
        for seg in merged {
            let start = formatTimestamp(seg.start)
            let end = formatTimestamp(seg.end)
            let text = seg.text.trimmingCharacters(in: .whitespaces)
            guard !text.isEmpty else { continue }
            lines.append("[\(start) - \(end)] \(seg.speaker): \(text)")
        }

        // No token timings: emit the plain transcript rather than an empty file
        let plainText = asrResult.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if lines.isEmpty && !plainText.isEmpty {
            FileHandle.standardError.write(Data("Warning: no word timings returned; speaker labels unavailable\n".utf8))
            lines.append("[\(formatTimestamp(0)) - \(formatTimestamp(audioDuration))] Unknown: \(plainText)")
        }

        // Print transcript
        print("")
        print("--- Transcript ---")
        print("")
        for line in lines {
            print(line)
            print("")
        }

        // Save to file
        let content = lines.joined(separator: "\n\n") + "\n"
        try content.write(toFile: outputPath, atomically: true, encoding: .utf8)

        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        let speakerCount = Set(diarizationResult.segments.map { $0.speakerId }).count
        print("")
        print("--- Summary ---")
        print(String(format: "Audio duration:  %.1fs (%.1f min)", audioDuration, audioDuration / 60))
        print(String(format: "Total time:      %.1fs (%.1f min)", totalTime, totalTime / 60))
        print(String(format: "Speed:           %.1fx realtime", Double(audioDuration) / totalTime))
        print("Speakers:        \(speakerCount)")
        print("Segments:        \(lines.count)")
        print("Saved to:        \(outputPath)")
    }

    func formatTimestamp(_ seconds: Float) -> String {
        let totalSeconds = Int(seconds)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, secs)
        }
        return String(format: "%02d:%02d", minutes, secs)
    }

    func validateOutputPath(_ path: String) throws {
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue {
            FileHandle.standardError.write(Data("Error: output path is a directory: \(path)\n".utf8))
            throw ExitCode.validationFailure
        }
        let parent = (path as NSString).deletingLastPathComponent
        guard FileManager.default.fileExists(atPath: parent, isDirectory: &isDir), isDir.boolValue,
              FileManager.default.isWritableFile(atPath: parent) else {
            FileHandle.standardError.write(Data("Error: output directory missing or not writable: \(parent)\n".utf8))
            throw ExitCode.validationFailure
        }
    }

    /// Merge consecutive same-speaker words, but start a new line after a pause
    /// so a monologue does not collapse into one entry.
    func mergeConsecutive(
        _ segments: [(speaker: String, start: Float, end: Float, text: String)],
        maxGap: Float = 1.5
    ) -> [(speaker: String, start: Float, end: Float, text: String)] {
        var merged: [(speaker: String, start: Float, end: Float, text: String)] = []
        for seg in segments {
            if let last = merged.last, last.speaker == seg.speaker, seg.start - last.end <= maxGap {
                merged[merged.count - 1].end = seg.end
                merged[merged.count - 1].text += " " + seg.text
            } else {
                merged.append(seg)
            }
        }
        return merged
    }
}

enum PipelineResult: Sendable {
    case transcription(ASRResult)
    case diarization(DiarizationResult)
}

enum TranscribeError: Error, CustomStringConvertible {
    case pipelineFailed

    var description: String {
        switch self {
        case .pipelineFailed: return "Transcription or diarization pipeline failed"
        }
    }
}
