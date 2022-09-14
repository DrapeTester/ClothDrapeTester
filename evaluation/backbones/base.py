import torch.nn as nn
import torch
torch.manual_seed(0)

class BaseModel(nn.Module):
    def __init__(self, model_name, input_mean, input_std, output_mean,
                 output_std, num_of_input, num_of_output):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.num_of_input = num_of_input
        self.num_of_output = num_of_output

        self.input_mean = torch.from_numpy(input_mean)
        self.input_std = torch.from_numpy(input_std)
        self.output_mean = torch.from_numpy(output_mean)
        self.output_std = torch.from_numpy(output_std)

        assert len(output_mean) == len(output_std) and (num_of_output == 3
                                                        or num_of_output == 6)
        if num_of_output == 6:
            self.lower_bound = torch.tensor([[0, 0, 0, -100, -100, -100]])
            self.upper_bound = torch.tensor([[100, 100, 100, 100, 100, 100]])
        elif num_of_output == 3:
            self.lower_bound = torch.tensor([[0, 0, 0]])
            self.upper_bound = torch.tensor([[100, 100, 100]])
        else:
            raise ValueError("the output size cannot be determined")

    def show_parameters(self):
        for name, param in self.named_parameters():
            print(
                f"[debug] {name} {param.shape} , can be changed = {param.requires_grad}"
            )

    def print_num_of_params(self):
        return

    def normalize_input(self, x):

        if self.input_mean is not None and self.input_std is not None:
            # print(
            #     f"[normalize] {x.shape} {self.input_mean.shape} {self.input_std.shape}"
            # )
            return (x - self.input_mean) / self.input_std
        else:
            return x

    def unnormalize_output(self, x):
        if self.output_std is not None and self.output_mean is not None:
            # print(
            #     f"[unnormalize] {x.shape} {self.input_mean.shape} {self.input_std.shape}"
            # )
            return x * self.output_std + self.output_mean
        else:
            return x

    def to(self, device):
        super(BaseModel, self).to(device)
        self.input_mean = self.input_mean.to(device)
        self.input_std = self.input_std.to(device)
        self.output_mean = self.output_mean.to(device)
        self.output_std = self.output_std.to(device)
        self.lower_bound = self.lower_bound.to(device)
        self.upper_bound = self.upper_bound.to(device)
        return self

    def clamp(self, x):
        if self.training == False:
            x = torch.max(torch.min(x, self.upper_bound), self.lower_bound)
        return x