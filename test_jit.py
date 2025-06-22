import torch
import librosa


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


def add():

    model = MyModel() 
    x = torch.randn(1, 10)
    traced_model = torch.jit.trace(model, x)
    traced_model.save("model_traced.pt")
    loaded_model = torch.jit.load("model_traced.pt")
    output = loaded_model(torch.randn(3, 10))
    print(output)


def add2():

    import torch
    import librosa
    import matplotlib.pyplot as plt

    # Load audio
    audio_path = "./assets/Mojito.mp3"
    audio, _ = librosa.load(path=audio_path, sr=16000, mono=True)
    sr = 16000
    device = "cuda"

    x = torch.Tensor(audio)[None, None, 10 * sr : 12 * sr].to(device)

    loaded_model = torch.jit.load("model_traced_slakh.pt").to(device)
    output_dict = loaded_model(x)

    # Plot and visualize
    frame_roll = output_dict["frame_roll"].data.cpu().numpy()
    onset_roll = output_dict["onset_roll"].data.cpu().numpy()
    offset_roll = output_dict["offset_roll"].data.cpu().numpy()
    drum_roll = output_dict["drum_roll"].data.cpu().numpy()
    program_roll = output_dict["program_roll"].data.cpu().numpy()

    fig, axs = plt.subplots(5, 1, sharex=True)
    axs[0].matshow(frame_roll.T, origin='lower', aspect='auto', cmap='jet')
    axs[1].matshow(onset_roll.T, origin='lower', aspect='auto', cmap='jet')
    axs[2].matshow(offset_roll.T, origin='lower', aspect='auto', cmap='jet')
    axs[3].matshow(drum_roll.T, origin='lower', aspect='auto', cmap='jet')
    axs[4].matshow(program_roll.T, origin='lower', aspect='auto', cmap='jet')
    plt.savefig("_zz.pdf")


if __name__ == '__main__':

    add2()