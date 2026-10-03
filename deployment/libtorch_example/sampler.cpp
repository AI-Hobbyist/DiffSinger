// Standalone shallow acoustic sampler smoke example. Inputs are synthetic;
// an application must obtain condition/aux_mel from the FS2 + auxiliary stage.
#include <torch/script.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: diffsinger_sampler acoustic.diffusion.pt hidden mel_bins frames [cuda]\n";
        return 1;
    }
    try {
        const auto device = torch::Device(argc > 5 ? torch::kCUDA : torch::kCPU);
        torch::NoGradGuard guard;
        auto module = torch::jit::load(argv[1], device);
        module.eval();
        const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const int64_t hidden = std::stoll(argv[2]);
        const int64_t mel_bins = std::stoll(argv[3]);
        const int64_t frames = std::stoll(argv[4]);
        auto condition = torch::zeros({1, frames, hidden}, options);
        auto auxiliary = torch::zeros({1, frames, mel_bins}, options);
        auto depth = torch::tensor(0.1, options);
        std::vector<torch::jit::IValue> inputs{condition, auxiliary, depth, int64_t(5)};
        auto mel = module.forward(inputs).toTensor();
        std::cout << mel.sizes() << '\n';
        return 0;
    } catch (const c10::Error& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
