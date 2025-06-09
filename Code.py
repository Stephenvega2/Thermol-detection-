import numpy as np
from scipy.stats import entropy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics import Color, Line, Rectangle
from kivy.clock import Clock
from kivy.properties import NumericProperty, ListProperty, StringProperty, ObjectProperty
from kivy.logger import Logger
from kivy.graphics.texture import Texture
import secrets
import random
import json
import time
import os
import pygame
from flask import Flask, jsonify, request
from kivy.utils import platform
import tempfile

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Flask setup
app = Flask(__name__)

# Simulated environment objects
class SimulatedObject:
    def __init__(self, x, y, speed, size):
        self.x = x
        self.y = y
        self.speed = speed
        self.size = size
        self.temp = 25  # Base temperature

objects = [
    SimulatedObject(100, 100, 2, 30),
    SimulatedObject(200, 200, 1.5, 40),
    SimulatedObject(300, 300, 2.5, 35)
]

class GroundingWidget(BoxLayout):
    frame = NumericProperty(0)
    input_load_data = ListProperty([])
    sys_load_data = ListProperty([])
    temp_data = ListProperty([])
    entropy_data = ListProperty([])
    energy_data = ListProperty([])
    efficiency_data = ListProperty([])
    detection_rate_data = ListProperty([])
    evasion_rate_data = ListProperty([])
    sensor_temp_data = ListProperty([])
    status_text = StringProperty("Simulation Running")
    detection_text = StringProperty("Detection: Waiting...")
    pygame_surface = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Simulation parameters
        self.surge_load = 10000
        self.base_load = 100
        self.ground_factor = 0.1
        self.damping = 0.01
        self.t_max = 0.01
        self.dt = 0.00005
        self.t_start = 0.001
        self.thermal_mass = 0.05
        self.heat_coeff = 0.001
        self.dissipation_coeff = 0.1
        self.initial_temp = 25
        self.ambient_temp = 25
        self.max_load = 1e4
        self.max_temp = 100
        self.n_bins = 50
        self.window_size = 100
        self.energy = 0
        self.use_device = PSUTIL_AVAILABLE and platform != 'android'
        self.device_load = self.base_load
        self.cycle_count = 0
        self.max_cycles = 3
        self.detection_threshold = 40  # °C
        self.evasion_factor = 0.1
        self.sensor_temp_data = []
        # Pygame setup (integrated with Kivy)
        pygame.init()
        self.screen = pygame.Surface((400, 400))  # Off-screen surface
        # Results file
        if platform == 'android':
            self.results_file = os.path.join(tempfile.gettempdir(), f"simulation_results_{int(time.time())}.json")
        else:
            self.results_file = f"simulation_results_{int(time.time())}.json"
        self.results_data = []
        self.save_header()
        # Initialize UI
        self.orientation = 'vertical'
        # Simulated environment display
        self.env_widget = Image(size_hint=(1, 0.4))
        self.add_widget(self.env_widget)
        # Detection results display
        self.detection_box = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        self.detection_label = Label(
            text="[b]Detection Status:[/b] [color=000000]Waiting...[/color]",
            markup=True,
            size_hint=(1, 1)
        )
        self.detection_box.add_widget(self.detection_label)
        self.add_widget(self.detection_box)
        # Status label
        self.label = Label(
            text="[b]Status:[/b] [color=000000]Simulation Running[/color]",
            markup=True,
            size_hint=(1, 0.1)
        )
        self.add_widget(self.label)
        # Plot widget
        self.plot_widget = BoxLayout(size_hint=(1, 0.3))
        self.add_widget(self.plot_widget)
        # Reset button
        self.reset_button = Button(text="Reset Simulation", size_hint=(1, 0.1))
        self.reset_button.bind(on_press=self.reset_simulation)
        self.add_widget(self.reset_button)
        # Initialize simulation
        self.initialize_simulation()
        Clock.schedule_interval(self.update, 1/30)

    def save_header(self):
        try:
            self.results_data = []
            with open(self.results_file, 'w') as f:
                json.dump(self.results_data, f, indent=4)
        except Exception as e:
            self.status_text = f"Error initializing JSON file: {str(e)}"
            Logger.error(f"GroundingWidget: Error initializing JSON file: {str(e)}")

    def initialize_simulation(self):
        self.time = np.arange(0, self.t_max, self.dt)
        self.input_load = np.where(self.time >= self.t_start, self.surge_load, self.base_load)
        self.sys_load = np.zeros_like(self.time)
        self.temp = np.zeros_like(self.time)
        self.entropy_vals = np.zeros_like(self.time)
        self.energy_vals = np.zeros_like(self.time)
        self.efficiency_vals = np.zeros_like(self.time)
        self.detection_rate_data = np.zeros_like(self.time)
        self.evasion_rate_data = np.zeros_like(self.time)
        self.sensor_temp_data = np.zeros_like(self.time)
        self.temp[0] = self.initial_temp
        self.energy = 0

        for i in range(1, len(self.time)):
            if self.use_device and PSUTIL_AVAILABLE:
                try:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.device_load = self.base_load + (cpu_percent / 100) * (self.surge_load - self.base_load)
                    self.input_load[i] = self.device_load if self.time[i] < self.t_start else self.surge_load
                    self.sensor_temp_data[i] = self.temp[i-1] + random.uniform(-0.5, 0.5)
                except Exception as e:
                    self.status_text = f"Error accessing CPU data: {str(e)}. Using simulated thermal data."
                    Logger.warning(f"GroundingWidget: {str(e)}. Falling back to simulated thermal data.")
                    self.use_device = False
            if not self.use_device:
                self.device_load = self.base_load + (secrets.randbelow(101) - 50)
                self.input_load[i] = self.device_load if self.time[i] < self.t_start else self.surge_load
                self.sensor_temp_data[i] = self.temp[i-1] + random.uniform(-0.5, 0.5)
            excess_load = self.input_load[i] - self.base_load
            grounded_load = excess_load * self.ground_factor
            self.sys_load[i] = self.base_load + grounded_load
            self.sys_load[i] = self.sys_load[i-1] + self.dt * (self.sys_load[i] - self.sys_load[i-1]) / self.damping
            self.sys_load[i] = np.clip(self.sys_load[i], 0, self.max_load)
            power = min(self.sys_load[i] * self.heat_coeff, 1e6)
            self.temp[i] = self.temp[i-1] + self.dt * (
                (power - self.dissipation_coeff * (self.temp[i-1] - self.ambient_temp)) / self.thermal_mass
            )
            self.temp[i] = np.clip(self.temp[i], self.ambient_temp, self.max_temp)
            self.energy += power * self.dt
            self.energy_vals[i] = self.energy
            self.efficiency_vals[i] = grounded_load / self.input_load[i] if self.input_load[i] > 0 else 0
            if i >= self.window_size:
                window = self.sys_load[i-self.window_size:i]
                hist, _ = np.histogram(window, bins=self.n_bins, range=(0, self.max_load), density=True)
                hist = hist + 1e-10
                self.entropy_vals[i] = entropy(hist, base=2)
            detection_sensitivity = 1 - self.sys_load[i] / self.max_load
            self.detection_rate_data[i], self.evasion_rate_data[i] = self.compute_detection_and_evasion(
                self.sensor_temp_data[i], detection_sensitivity, self.entropy_vals[i]
            )

        self.input_load_data = list(self.input_load)
        self.sys_load_data = list(self.sys_load)
        self.temp_data = list(self.temp)
        self.entropy_data = list(self.entropy_vals)
        self.energy_data = list(self.energy_vals)
        self.efficiency_data = list(self.efficiency_vals)
        self.detection_rate_data = list(self.detection_rate_data)
        self.evasion_rate_data = list(self.evasion_rate_data)
        self.sensor_temp_data = list(self.sensor_temp_data)
        self.max_frames = len(self.time)
        self.frame = 0
        self.save_results()

    def compute_detection_and_evasion(self, sensor_temp, detection_sensitivity, entropy):
        detection_signal = 1 if sensor_temp > self.detection_threshold * detection_sensitivity else 0
        evasion = 0
        if detection_signal:
            self.evasion_factor = min(0.9, self.evasion_factor + 0.01 * (entropy / np.log2(self.n_bins)))
            if random.random() < self.evasion_factor:
                detection_signal = 0
                evasion = 1
        detection_rate = detection_signal
        evasion_rate = evasion
        return detection_rate, evasion_rate

    def update_simulated_environment(self):
        self.screen.fill((0, 0, 50))  # Dark blue background
        current_sensor_temp = self.sensor_temp_data[int(self.frame)] if self.frame < len(self.sensor_temp_data) else self.ambient_temp
        intensity = np.clip((current_sensor_temp - self.ambient_temp) / (self.max_temp - self.ambient_temp), 0, 1)

        for obj in objects:
            obj.x += obj.speed
            if obj.x > 400: obj.x = 0
            temp_factor = (obj.temp - self.ambient_temp) / (self.max_temp - self.ambient_temp)
            color = (0, int(255 * temp_factor * intensity), int(255 * (1 - temp_factor) * intensity))
            pygame.draw.circle(self.screen, color, (int(obj.x), int(obj.y)), obj.size)

            # Simulate detection
            if (abs(obj.x - 200) < 50 and abs(obj.y - 200) < 50 and current_sensor_temp > self.detection_threshold):
                pygame.draw.rect(self.screen, (255, 0, 0), (int(obj.x - obj.size), int(obj.y - obj.size), obj.size * 2, obj.size * 2), 2)

        # Convert Pygame surface to Kivy texture
        data = pygame.image.tostring(self.screen, "RGB")
        texture = Texture.create(size=(400, 400), colorfmt='rgb')
        texture.blit_buffer(data, colorfmt='rgb', bufferfmt='ubyte')
        self.env_widget.texture = texture

    def update(self, dt):
        self.frame = min(self.frame + 1, self.max_frames - 1)
        if self.frame >= self.max_frames - 1 and self.cycle_count < self.max_cycles:
            self.cycle_count += 1
            self.initialize_simulation()

        # Update simulated environment
        self.update_simulated_environment()

        # Update detection text
        current_detection = self.detection_rate_data[int(self.frame)]
        current_evasion = self.evasion_rate_data[int(self.frame)]
        current_sensor_temp = self.sensor_temp_data[int(self.frame)]
        detection_status = "Detected" if current_detection > 0 else "Not Detected"
        evasion_status = "Evaded" if current_evasion > 0 else "Not Evaded"
        self.detection_text = (
            f"[b]Detection:[/b] {detection_status} | "
            f"[b]Evasion:[/b] {evasion_status} | "
            f"[b]Sensor Temp:[/b] {current_sensor_temp:.1f}°C"
        )
        self.detection_label.text = f"[color=000000]{self.detection_text}[/color]"
        self.label.text = f"[b]Status:[/b] [color=000000]{self.status_text}[/color]"

        # Update plots
        self.plot_widget.canvas.before.clear()
        with self.plot_widget.canvas.before:
            Color(1, 1, 1, 1)
            Rectangle(pos=self.plot_widget.pos, size=self.plot_widget.size)
            Color(0, 0, 0, 1)
            plot_height = self.plot_widget.height / 8
            for i in range(8):
                y = self.plot_widget.height - (i + 1) * plot_height
                Line(points=[50, y, self.plot_widget.width - 50, y], width=2)
                Line(points=[50, y - plot_height + 20, 50, y], width=2)
                for j in range(1, 5):
                    y_grid = y - (j * (plot_height - 20) / 5)
                    Line(points=[50, y_grid, self.plot_widget.width - 50, y_grid], width=1, dash_length=5, dash_offset=5)
            max_load_val = self.surge_load * 1.1
            max_T = max(self.temp_data) * 1.1 if max(self.temp_data) > self.initial_temp else self.initial_temp + 10
            max_entropy = max(self.entropy_data) * 1.1 if max(self.entropy_data) > 0 else np.log2(self.n_bins)
            max_energy = max(self.energy_data) * 1.1 if max(self.energy_data) > 0 else 100
            max_efficiency = max(self.efficiency_data) * 1.1 if max(self.efficiency_data) > 0 else 1
            max_detection = 1.1
            max_evasion = 1.1
            max_sensor = max(self.sensor_temp_data) * 1.1 if max(self.sensor_temp_data) > 0 else 50
            for i in range(int(self.frame)):
                x = 50 + (i / self.max_frames) * (self.plot_widget.width - 100)
                y_input = self.plot_widget.height - plot_height - (self.input_load_data[i] / max_load_val) * (plot_height - 20)
                y_sys = self.plot_widget.height - plot_height - (self.sys_load_data[i] / max_load_val) * (plot_height - 20)
                if i > 0:
                    Color(1, 0, 0, 1)
                    Line(points=[x_prev, y_input_prev, x, y_input], width=2)
                    Color(0, 0, 1, 1)
                    Line(points=[x_prev, y_sys_prev, x, y_sys], width=2)
                x_prev, y_input_prev, y_sys_prev = x, y_input, y_sys
                y_temp = self.plot_widget.height - 2 * plot_height - ((self.temp_data[i] - self.ambient_temp) / (max_T - self.ambient_temp)) * (plot_height - 20)
                if i > 0:
                    Color(0, 1, 0, 1)
                    Line(points=[x_prev_t, y_temp_prev, x, y_temp], width=2)
                x_prev_t, y_temp_prev = x, y_temp
                y_entropy = self.plot_widget.height - 3 * plot_height - (self.entropy_data[i] / max_entropy) * (plot_height - 20)
                if i > 0:
                    Color(0.5, 0, 0.5, 1)
                    Line(points=[x_prev_e, y_entropy_prev, x, y_entropy], width=2)
                x_prev_e, y_entropy_prev = x, y_entropy
                y_energy = self.plot_widget.height - 4 * plot_height - (self.energy_data[i] / max_energy) * (plot_height - 20)
                if i > 0:
                    Color(1, 0.5, 0, 1)
                    Line(points=[x_prev_en, y_energy_prev, x, y_energy], width=2)
                x_prev_en, y_energy_prev = x, y_energy
                y_efficiency = self.plot_widget.height - 5 * plot_height - (self.efficiency_data[i] / max_efficiency) * (plot_height - 20)
                if i > 0:
                    Color(0, 0.5, 0.5, 1)
                    Line(points=[x_prev_eff, y_efficiency_prev, x, y_efficiency], width=2)
                x_prev_eff, y_efficiency_prev = x, y_efficiency
                y_detection = self.plot_widget.height - 6 * plot_height - (self.detection_rate_data[i] / max_detection) * (plot_height - 20)
                if i > 0:
                    Color(0.2, 0.8, 0.2, 1)
                    Line(points=[x_prev_det, y_detection_prev, x, y_detection], width=2)
                x_prev_det, y_detection_prev = x, y_detection
                y_evasion = self.plot_widget.height - 7 * plot_height - (self.evasion_rate_data[i] / max_evasion) * (plot_height - 20)
                if i > 0:
                    Color(0.8, 0.2, 0.2, 1)
                    Line(points=[x_prev_eva, y_evasion_prev, x, y_evasion], width=2)
                x_prev_eva, y_evasion_prev = x, y_evasion
                y_sensor = self.plot_widget.height - 8 * plot_height - (self.sensor_temp_data[i] / max_sensor) * (plot_height - 20)
                if i > 0:
                    Color(0, 0.5, 1, 1)
                    Line(points=[x_prev_sen, y_sensor_prev, x, y_sensor], width=2)
                x_prev_sen, y_sensor_prev = x, y_sensor
            Color(0, 0, 0, 1)
            self.plot_widget.canvas.after.clear()
            with self.plot_widget.canvas.after:
                Label(text="Load (tasks/s)", pos=(0, self.plot_widget.height - plot_height / 2), size=(50, 20))
                Label(text="Temp (°C)", pos=(0, self.plot_widget.height - 3 * plot_height / 2), size=(50, 20))
                Label(text="Entropy (bits)", pos=(0, self.plot_widget.height - 5 * plot_height / 2), size=(50, 20))
                Label(text="Energy (J)", pos=(0, self.plot_widget.height - 7 * plot_height / 2), size=(50, 20))
                Label(text="Efficiency", pos=(0, self.plot_widget.height - 9 * plot_height / 2), size=(50, 20))
                Label(text="Detection Rate", pos=(0, self.plot_widget.height - 11 * plot_height / 2), size=(50, 20))
                Label(text="Evasion Rate", pos=(0, self.plot_widget.height - 13 * plot_height / 2), size=(50, 20))
                Label(text="Sensor Temp (°C)", pos=(0, self.plot_widget.height - 15 * plot_height / 2), size=(50, 20))

    def save_results(self):
        try:
            for i in range(len(self.time)):
                self.results_data.append({
                    'Cycle': self.cycle_count,
                    'Time': float(self.time[i]),
                    'Input Load (tasks/s)': float(self.input_load[i]),
                    'System Load (tasks/s)': float(self.sys_load[i]),
                    'Temperature (°C)': float(self.temp[i]),
                    'Entropy (bits)': float(self.entropy_vals[i]),
                    'Energy (J)': float(self.energy_vals[i]),
                    'Efficiency': float(self.efficiency_vals[i]),
                    'Detection Rate': float(self.detection_rate_data[i]),
                    'Evasion Rate': float(self.evasion_rate_data[i]),
                    'Sensor Temp (°C)': float(self.sensor_temp_data[i])
                })
            with open(self.results_file, 'w') as f:
                json.dump(self.results_data, f, indent=4)
            self.status_text = (
                f"Cycle {self.cycle_count} | Max Temp: {max(self.temp_data):.1f}°C | "
                f"Avg Entropy: {np.mean(self.entropy_vals):.2f} bits | "
                f"Total Energy: {self.energy:.2f} J | "
                f"Det Rate: {self.detection_rate_data[-1]:.2%} | "
                f"Eva Rate: {self.evasion_rate_data[-1]:.2%} | "
                f"Sensor Temp: {self.sensor_temp_data[-1]:.1f}°C"
            )
            Logger.info(f"GroundingWidget: Saved results. Sensor Temp: {self.sensor_temp_data[-1]:.1f}°C")
        except Exception as e:
            self.status_text = f"Error saving results to JSON: {str(e)}"
            Logger.error(f"GroundingWidget: Error saving results to JSON: {str(e)}")

    def reset_simulation(self, instance):
        self.cycle_count = 0
        self.frame = 0
        self.energy = 0
        self.evasion_factor = 0.1
        self.results_data = []
        self.initialize_simulation()
        self.status_text = "Simulation Reset"
        self.detection_text = "Detection: Waiting..."
        Logger.info("GroundingWidget: Simulation reset")

    def on_stop(self):
        pygame.quit()

# Flask routes
@app.route('/status', methods=['GET'])
def get_status():
    widget = GroundingWidget()
    return jsonify({
        'status': widget.status_text,
        'detection': widget.detection_text,
        'sensor_temp': widget.sensor_temp_data[-1] if widget.sensor_temp_data else 25
    })

@app.route('/reset', methods=['POST'])
def reset():
    widget = GroundingWidget()
    widget.reset_simulation(None)
    return jsonify({'message': 'Simulation reset'})

class GroundingApp(App):
    def build(self):
        return GroundingWidget()

if __name__ == '__main__':
    # Run Flask with error handling for dotenv
    import threading
    try:
        import dotenv
    except ImportError:
        Logger.warning("python-dotenv not installed. Proceeding without .env support.")
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=True), daemon=True).start()
    GroundingApp().run()
