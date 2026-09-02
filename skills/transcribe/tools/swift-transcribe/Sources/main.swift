import ArgumentParser
import Foundation
import WhisperKit
import SpeakerKit

@main
struct TranscribeTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "swift-transcribe",
        abstract: "Transcribe audio with speaker diarization using WhisperKit + SpeakerKit"
    )

    @Argument(help: "Path to the audio file")
    var audioPath: String

    static let defaultModel = "large-v3-v20240930"

    @Option(name: .long, help: "Whisper model to use (default: large-v3-turbo)")
    var model: String = Self.defaultModel

    @Option(name: .long, help: "Language code (e.g. ja, fr). Auto-detected if omitted")
    var language: String?

    @Option(name: .long, help: "Number of speakers (auto-detect if omitted)")
    var numSpeakers: Int?

    @Flag(name: .long, help: "Use distil-large-v3 for faster transcription")
    var fast: Bool = false

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

        if fast && model != Self.defaultModel {
            FileHandle.standardError.write(Data("Error: --fast and --model are mutually exclusive\n".utf8))
            throw ExitCode.validationFailure
        }
        if let n = numSpeakers, n < 1 {
            FileHandle.standardError.write(Data("Error: --num-speakers must be at least 1\n".utf8))
            throw ExitCode.validationFailure
        }
        let selectedModel = fast ? "distil-whisper_distil-large-v3" : "openai_whisper-\(model)"

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
        print("Model: \(selectedModel)")
        print("")

        // Load audio once, shared by both pipelines. This holds the whole file as
        // Float32 (64 KB per second) instead of using the libraries' streaming
        // URL overloads; fine for hours of audio, not for days.
        print("Loading audio...")
        let loadStart = CFAbsoluteTimeGetCurrent()
        let audioArray = try AudioProcessor.loadAudioAsFloatArray(fromPath: resolvedPath)
        guard !audioArray.isEmpty else {
            FileHandle.standardError.write(Data("Error: no audio samples decoded from \(fileName)\n".utf8))
            throw ExitCode.failure
        }
        let audioDuration = Float(audioArray.count) / 16000.0
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        print(String(format: "Audio loaded: %.1fs duration (%.1fs to load)", audioDuration, loadTime))
        print("")

        // Run transcription and diarization in parallel
        print("Starting parallel transcription + diarization...")
        print("")

        let transcriptionResult: [TranscriptionResult]
        let diarizationResult: DiarizationResult

        // Use a task group for true parallelism
        (transcriptionResult, diarizationResult) = try await withThrowingTaskGroup(
            of: PipelineResult.self
        ) { group in
            // Transcription task
            group.addTask {
                let tStart = CFAbsoluteTimeGetCurrent()
                print("[Transcription] Initializing WhisperKit...")

                let config = WhisperKitConfig(
                    model: selectedModel,
                    verbose: verbose,
                    logLevel: verbose ? .debug : .error
                )
                let whisper = try await WhisperKit(config)

                print("[Transcription] Model loaded, transcribing...")
                var decodeOptions = DecodingOptions()
                decodeOptions.wordTimestamps = true
                if let language {
                    decodeOptions.language = language
                    decodeOptions.detectLanguage = false
                }

                let results = try await whisper.transcribe(
                    audioArray: audioArray,
                    decodeOptions: decodeOptions
                )

                let tTime = CFAbsoluteTimeGetCurrent() - tStart
                let segmentCount = results.reduce(0) { $0 + $1.segments.count }
                print(String(format: "[Transcription] Complete: %d segments in %.1fs (%.1fx realtime)",
                             segmentCount, tTime, Double(audioDuration) / tTime))

                await whisper.unloadModels()
                return .transcription(results)
            }

            // Diarization task
            group.addTask {
                let dStart = CFAbsoluteTimeGetCurrent()
                print("[Diarization] Initializing SpeakerKit...")

                let pyConfig = PyannoteConfig(verbose: verbose)
                let diarizer = try await SpeakerKit(pyConfig)

                print("[Diarization] Models loaded, diarizing...")
                var options = PyannoteDiarizationOptions()
                if let n = numSpeakers {
                    options.numberOfSpeakers = n
                }

                let result = try await diarizer.diarize(
                    audioArray: audioArray,
                    options: options
                )

                let dTime = CFAbsoluteTimeGetCurrent() - dStart
                print(String(format: "[Diarization] Complete: %d speakers in %.1fs (%.1fx realtime)",
                             result.speakerCount, dTime, Double(audioDuration) / dTime))

                await diarizer.unloadModels()
                return .diarization(result)
            }

            var transcription: [TranscriptionResult]?
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
        let speakerSegments = diarizationResult.addSpeakerInfo(
            to: transcriptionResult,
            strategy: .subsegment
        )

        // Flatten and merge consecutive same-speaker segments
        var flat: [(speaker: String, start: Float, end: Float, text: String)] = []
        for group in speakerSegments {
            for segment in group {
                let speaker = formatSpeaker(segment.speaker)
                let text = segment.text.trimmingCharacters(in: .whitespaces)
                guard !text.isEmpty else { continue }
                flat.append((speaker, segment.startTime, segment.endTime, text))
            }
        }

        let merged = mergeConsecutive(flat)

        // Format output lines
        var lines: [String] = []
        for seg in merged {
            let start = formatTimestamp(seg.start)
            let end = formatTimestamp(seg.end)
            lines.append("[\(start) - \(end)] \(seg.speaker): \(seg.text)")
        }

        // No word timings (model without alignment heads): SpeakerKit drops every
        // segment, so emit the plain transcript rather than an empty file.
        if lines.isEmpty {
            let plainText = transcriptionResult.map(\.text).joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !plainText.isEmpty {
                FileHandle.standardError.write(Data("Warning: no word timings returned; speaker labels unavailable\n".utf8))
                lines.append("[\(formatTimestamp(0)) - \(formatTimestamp(audioDuration))] Unknown: \(plainText)")
            }
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
        print("")
        print("--- Summary ---")
        print(String(format: "Audio duration:  %.1fs (%.1f min)", audioDuration, audioDuration / 60))
        print(String(format: "Total time:      %.1fs (%.1f min)", totalTime, totalTime / 60))
        print(String(format: "Speed:           %.1fx realtime", Double(audioDuration) / totalTime))
        print("Speakers:        \(diarizationResult.speakerCount)")
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

    /// Merge consecutive same-speaker segments, but start a new line after a pause
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

    func formatSpeaker(_ speaker: SpeakerInfo) -> String {
        switch speaker {
        case .speakerId(let id):
            return "Speaker \(id + 1)"
        case .multiple(let ids):
            let labels = ids.map { "Speaker \($0 + 1)" }
            return labels.joined(separator: "/")
        case .noMatch:
            return "Unknown"
        @unknown default:
            return "Unknown"
        }
    }
}

enum PipelineResult: Sendable {
    case transcription([TranscriptionResult])
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
