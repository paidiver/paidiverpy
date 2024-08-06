import yaml
from pathlib import Path
import json


class DynamicConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key.endswith('_path'):
                value = Path(value)
            setattr(self, key, value)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self, convert_path=True):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                if convert_path:
                    result[key] = str(value)
                else:
                    result[key] = value
            elif isinstance(value, DynamicConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

class GeneralConfig(DynamicConfig):
    pass

class PositionConfig(DynamicConfig):
    pass

class ConvertConfig(DynamicConfig):
    pass

class ColorConfig(DynamicConfig):
    pass

class SamplingConfig(DynamicConfig):
    pass

class EdgeConfig(DynamicConfig):
    pass

config_class_mapping = {
    'general': GeneralConfig,
    'position': PositionConfig,
    'sampling': SamplingConfig,
    'edge': EdgeConfig,
    'color': ColorConfig,
    'convert': ConvertConfig
}


class Configuration:
    def __init__(self, config_file_path=None, input_path=None, output_path=None):
        self.general = None
        self.position = None
        self.sampling = None
        self.edge = None
        self.preprocessing = None
        self.convert = None
        self.steps = []

        if config_file_path:
            self._load_config_from_file(config_file_path)
        else:
            self._validate_paths(input_path, output_path)
            self.general = GeneralConfig(input_path=input_path, output_path=output_path)

    def _load_config_from_file(self, config_file_path):
        try:
            with open(config_file_path, "r", encoding='utf-8') as config_file:
                config_data = yaml.safe_load(config_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to load the configuration file: {str(e)}") from e
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to load the configuration file: {str(e)}") from e

        self.general = self._validate_general_config(config_data)
        self._load_steps(config_data)

    def _validate_general_config(self, config_data):
        if 'general' not in config_data:
            raise ValueError("General configuration is not specified.")
        if not config_data['general']:
            raise ValueError("General configuration is empty.")
        if not config_data['general'].get('input_path') or not config_data['general'].get('output_path'):
            raise ValueError("Input and output paths are not specified.")
        name = config_data['general'].get('name')
        if not name:
            config_data['general']['name'] = "raw"
        config_data['general']['step_name'] = "Open"
        return GeneralConfig(**config_data['general'])

    def _validate_paths(self, input_path, output_path):
        if not input_path or not output_path:
            raise ValueError("Input and output paths are not specified.")

    def _load_steps(self, config_data):
        if 'steps' in config_data and config_data['steps']:
            for step_order, step in enumerate(config_data['steps']):
                for step_name, step_config in step.items():
                    if step_name in config_class_mapping:
                        name = step_config.get('name')
                        if not name:
                            step_config['name'] = f"{step_name}_{step_order + 1}"
                        step_config['step_name'] = step_name
                        step_class = config_class_mapping[step_name]
                        step_instance = step_class(**step_config)
                        self.steps.append(step_instance)
                    else:
                        raise ValueError(f"Invalid step name: {step_name}")

    def add_config(self, config_name, config):


        if config_name not in config_class_mapping:
            raise ValueError(f"Invalid configuration name: {config_name}")

        current_config = getattr(self, config_name)
        if current_config is None:
            setattr(self, config_name, config_class_mapping[config_name](**config))
        else:
            current_config.update(**config)

    def add_step(self, config_index, parameters):
        if len(self.steps) == 0:
            self.steps.append(config_class_mapping[parameters['step_name']](**parameters))
            return len(self.steps) - 1
        if config_index is None:
            self.steps.append(config_class_mapping[parameters['step_name']](**parameters))
            return len(self.steps) - 1
        elif config_index < len(self.steps):
            self.steps[config_index].update(**parameters)
            return config_index
        else:
            raise ValueError(f"Invalid step index: {config_index}")

    def to_dict(self):
        result = {}
        if self.general:
            result['general'] = self.general.to_dict()
        if self.position:
            result['position'] = self.position.to_dict()
        if self.sampling:
            result['sampling'] = self.sampling.to_dict()
        if self.edge:
            result['edge'] = self.edge.to_dict()
        if self.preprocessing:
            result['preprocessing'] = self.preprocessing.to_dict()
        if self.convert:
            result['convert'] = self.convert.to_dict()
        result['steps'] = [step.to_dict() for step in self.steps]
        return result

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4)



    # def to_html(self):
    #     dot = graphviz.Digraph(comment='Configuration')
    #     dot.node('general', 'General')

    #     for i, step in enumerate(self.steps):
    #         step_name = step.name if hasattr(step, 'name') else f'Step {i+1}'
    #         dot.node(f'step_{i}', step_name)
    #         dot.edge('general', f'step_{i}')

    #     # Layout adjustments (optional)
    #     dot.attr(rankdir='LR')  # Left to right layout

    #     return HTML(dot.pipe('svg'))



    # def to_html(self):
    #     steps_html = ""
    #     for i, step in enumerate(self.steps):
    #         if i % 4 == 0 and i > 0:
    #             steps_html += '<div style="clear:both;"></div>'
    #         steps_html += f"""
    #             <div style="float:left; width: 100px; height: 80px; margin: 10px; border: 1px solid #000; text-align: center; line-height: 80px;">
    #                 <h2 style="font-size:20px;">{step.name.capitalize()}</h2>
    #                 <h2 style="font-size:13px;">Type: {step.step_name.capitalize()}</h2>
    #             </div>
    #         """
    #         if i < len(self.steps) - 1:
    #             steps_html += """
    #                 <div style="float:left; width: 50px; height: 80px; margin: 10px; text-align: center; line-height: 80px;">
    #                     &#10132;
    #                 </div>
    #             """
        
    #     html = f"""
    #     <div style="display: flex; flex-wrap: wrap; align-items: center;">
    #         <div style="float:left; width: 100px; height: 80px; margin: 10px; border: 1px solid #000; text-align: center; line-height: 80px;">
    #             <h2 style="font-size:20px;">{getattr(self.general, 'name').capitalize()}</h2>
    #             <h2 style="font-size:13px;">Type: {getattr(self.general, 'step_name').capitalize()}</h2>
    #         </div>
    #         {f'<div style="float:left; width: 50px; height: 80px; margin: 10px; text-align: center; line-height: 80px;">&#10132;</div>{steps_html}' if self.steps else ''}
    #     </div>
    #     """
    #     return html

    # def _repr_html_(self):
    #     return self.to_html()
