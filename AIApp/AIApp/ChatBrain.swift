import Foundation

/// Swift side: load canned replies and route through Objective-C `BrainEngine` (Core ML).
final class ChatBrain: ObservableObject {
    private let engine = BrainEngine()

    /// True when `Brain.mlmodelc` and `brain_vocab.json` loaded successfully.
    @Published private(set) var isBrainReady: Bool

    private var responses: [String: String] = [:]

    init() {
        isBrainReady = false
        reloadResponsesFromBundle()
        isBrainReady = engine.isReady
    }

    func reloadResponsesFromBundle() {
        guard let url = Bundle.main.url(forResource: "brain_responses", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let intents = obj["intents"] as? [String: String] else {
            responses = [:]
            return
        }
        responses = intents
    }

    func reply(to userText: String) -> String {
        let trimmed = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return responses["greeting"] ?? "Say something!"
        }
        guard engine.isReady else {
            return "Brain didn’t load (missing Brain.mlmodelc or brain_vocab in the app bundle). "
                + "Rebuild after running train_brain.py, or install the IPA from CI."
        }
        let intent = engine.predictIntent(trimmed) ?? "greeting"
        if let body = responses[intent] {
            return body
        }
        return responses["greeting"] ?? "OK."
    }
}
