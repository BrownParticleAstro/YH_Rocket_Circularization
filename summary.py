import numpy as np
import matplotlib.pyplot as plt


class Summary:
    def __init__(self,):
        self.data = dict()

    def log(self, **kwargs):
        for key in kwargs:
            if key not in self.data:
                self.data[key] = [kwargs[key]]
            else:
                self.data[key].append(kwargs[key])

    def log_list(self, **kwargs):
        for key in kwargs:
            if key not in self.data:
                self.data[key] = kwargs[key]
            else:
                raise ValueError(f'The key {key} already exists in the list')

    def get_list(self, name, subindex=None):
        if subindex is None:
            return self.data[name]
        else:
            return [datum[subindex] for datum in self.data[name]]

    def get_rotation_transforms(self, positions):
        positions = self.data[positions]

        transforms = list()
        rs = list()
        thetas = list()
        for pos in positions:
            r = np.linalg.norm(pos)
            theta = np.arctan2(pos[1], pos[0])
            rhat = pos / r
            rot_mat = np.array([[rhat[0], -rhat[1]], [rhat[1], rhat[0]]])
            transforms.append(rot_mat)
            rs.append(r)
            thetas.append(theta)

        return transforms, rs, thetas

    def forward_transform(self, vecs, transform_name='transforms'):
        assert all(vec.shape == (2,) for vec in vecs)
        return [tr @ vec for tr, vec in zip(self.data[transform_name], self.data[vecs])]

    def inverse_transform(self, vecs, transform_name='transforms'):
        assert all(vec.shape == (2,) for vec in vecs)
        return [tr.T @ vec for tr, vec in zip(self.data[transform_name], self.data[vecs])]

    def get_angles(self, vecs):
        assert all(vec.shape == (2,) for vec in vecs)
        return [np.arctan2(vec[1], vec[0]) for vec in self.data[vecs]]

    def extract_process_save(self, inputs, function, outputs):
        raw_outputs = function(*inputs)
        for name, value in zip(outputs, raw_outputs):
            self.log_list({name: value})

        return raw_outputs

    def plot_values(self, ax, ys, labels, title=None, typ='Regular', x=None, max_index=None):
        if title is not None:
            ax.set_title(title)

        plot_types = {
            'Regular': ax.plot,
            'Log': ax.semilogy,
            'Loglog': ax.loglog
        }
        plot = plot_types[typ]

        if x is not None:
            x = self.data['x']
        else:
            x = range(len(self.data['states']))

        if isinstance(ys, str):
            ys = (self.data[ys],)
        elif isinstance(ys, list):
            ys = tuple(self.data[name] for name in ys)

        if max_index is not None:
            x, ys = x[:max_index], [y[:max_index] for y in ys]

        for y, label in zip(ys, labels):
            plot(x, y, label=label)

    def plot_summary(self, shape, logged_values, fig_size=(10, 5)):

        fig, axes = plt.subplots(*shape, figsize=fig_size, num=1, clear=True)
        fig.suptitle('Run Summary')

        for ax, config in zip(axes.flatten(), logged_values):
            self.plot_values(ax, **config)

        fig.tight_layout()

        return fig
