import torch
import torch.nn as nn
import torch.nn.functional as F


class NERFNet(nn.Module):
    def __init__(self):
        super(NERFNet, self).__init__()
        # Input sound features
        self.sound_fc = nn.Linear(sound_input_size, hidden_size)

        # Input microphone position features
        self.mic_fc = nn.Linear(3, hidden_size)

        # Input source position features
        self.source_fc = nn.Linear(3, hidden_size)

        # Fusion layer
        self.fusion_fc = nn.Linear(hidden_size * 3, fusion_size)

        # Decoder layers
        self.decoder_fc1 = nn.Linear(fusion_size, hidden_size)
        self.decoder_fc2 = nn.Linear(hidden_size, hidden_size)
        self.decoder_fc3 = nn.Linear(hidden_size, sound_output_size)

    def forward(self, sound, mic_position, source_position):
        # Process sound input
        sound_features = F.relu(self.sound_fc(sound))

        # Process microphone position input
        mic_features = F.relu(self.mic_fc(mic_position))

        # Process source position input
        source_features = F.relu(self.source_fc(source_position))

        # Concatenate features
        fusion_features = torch.cat(
            (sound_features, mic_features, source_features), dim=1)
        fusion_features = F.relu(self.fusion_fc(fusion_features))

        # Decoder layers
        x = F.relu(self.decoder_fc1(fusion_features))
        x = F.relu(self.decoder_fc2(x))
        output = self.decoder_fc3(x)

        return output
