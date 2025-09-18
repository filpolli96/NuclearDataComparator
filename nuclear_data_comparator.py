import uproot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
import ROOT
from pathlib import Path
from matplotlib import rcParams
import sympy as sp
import matplotlib.backends.backend_pdf

# Настройка стиля
plt.style.use('seaborn-v0_8-whitegrid')
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (10, 8)
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'

gamma_tolerance = 7.0

class NuclearDataComparator:
	def __init__(self):
		self.talys_data = None
		self.selected_element = None
<<<<<<< HEAD
=======
		self.sample_name = None
>>>>>>> d1da805 (Initial commit: Nuclear Data Comparator)
		self.talys_energies = set()
		self.target_neutron_energy = 14.1  # МэВ
		self.neutron_energy_tolerance = 2.0  # МэВ

        
		# Словарь для хранения соответствия авторов и стилей
		self.author_styles = {}

		# Доступные маркеры и цвета
		self.markers = ['s', '^', 'D', 'v', 'p', 'h', '8', 'X', 'P', '*', 'd', '>', '<']
		self.colors = [
			'#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
			'#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4',
			'#ff9896', '#98df8a', '#ffbb78', '#c5b0d5'
		]

		# Счетчики для назначения стилей
		self.marker_index = 0
		self.color_index = 0


	def set_element(self, element):
		"""Установка элемента для анализа"""
		self.selected_element = element
		print(f"Установлен элемент для анализа: {element}")
		return True

	def parse_talys_file(self, talys_file_path):
		"""Парсинг файла с данными TALYS"""
		data = []
		self.talys_energies = set()
        
		try:
			with open(talys_file_path, 'r') as f:
				next(f)  # Пропускаем заголовок
				for line in f:
					parts = line.strip().split()
					if len(parts) >= 7:
						try:
							energy_keV = float(parts[0])
							talys_entry = {
								'energy_keV': energy_keV,
								'mother_nucleus': parts[1],
								'reaction': parts[2],
								'final_nucleus': parts[3],
								'talys_cross_section': float(parts[4]),
								'abundance': float(parts[5]),
								'multipolarity': int(parts[6])
							}
							data.append(talys_entry)
							self.talys_energies.add(energy_keV)
						except ValueError:
							continue
		except Exception as e:
			print(f"Ошибка при чтении файла TALYS: {e}")
			return None
        
		self.talys_data = pd.DataFrame(data)
		print(f"Загружено {len(self.talys_data)} записей из TALYS")
		return self.talys_data

	#%%add_to NuclearDataComparator
	def is_energy_in_talys(self, energy_keV, tolerance=10):
		"""Проверка, есть ли энергия в данных TALYS"""
		if not self.talys_energies:
			return False
		
		closest_energy = min(self.talys_energies, key=lambda x: abs(x - energy_keV))
		#print("energy ", closest_energy, energy_keV)
		return abs(closest_energy - energy_keV) < tolerance

	#%%add_to NuclearDataComparator
	
	# not used
	def find_isotope_for_energy(self, energy_keV, tolerance=10):
		"""Поиск изотопа для заданной энергии гамма-квантов"""
		if self.talys_data is None or self.talys_data.empty:
			return None
		
		closest_match = None
		min_diff = float('inf')
		
		for _, row in self.talys_data.iterrows():
			diff = abs(row['energy_keV'] - energy_keV)
			if diff < min_diff and diff < tolerance:
				min_diff = diff
				closest_match = row
		
		# Возвращаем как словарь, а не pandas Series
		if closest_match is not None:
			return dict(closest_match)
		return None

	def find_all_talys_matches(self, energy_keV, tolerance=10.0):
		"""Поиск ВСЕХ близких энергий в TALYS данных независимо от изотопа"""
		all_matches = []
		
		if self.talys_data is not None:
			for _, row in self.talys_data.iterrows():
				talys_energy = row['energy_keV']
				diff = abs(talys_energy - energy_keV)
				
				if diff < tolerance:
					match_data = {
						'energy_keV': talys_energy,
						'cross_section': row['talys_cross_section'],
						'abundance': row['abundance'],
						'reaction': row['reaction'],
						'final_nucleus': row['final_nucleus'],
						'mother_nucleus': row['mother_nucleus'],
						'multipolarity': row['multipolarity'],
						'energy_diff': diff
					}
					all_matches.append(match_data)
		
		# Сортируем по близости к целевой энергии
		all_matches.sort(key=lambda x: x['energy_diff'])
		return all_matches
	
	def find_talys_graphs_for_all_matches(self, talys_matches, external_data_dir, tolerance=10.0):
		"""Поиск графиков TALYS для всех найденных совпадений"""
		talys_graphs_info = []
		
		for match in talys_matches:
			isotope = match['mother_nucleus']
			energy_keV = match['energy_keV']
			
			talys_ed_file = os.path.join(external_data_dir, "TalysED", "root", f"{isotope}.root")
			
			if os.path.exists(talys_ed_file):
				try:
					with uproot.open(talys_ed_file) as talys_file:
						best_graph = None
						min_diff = float('inf')
						best_talys_energy = 0
						
						for graph_name in talys_file.keys():
							energy_match = re.search(r'_EG_(\d+)', graph_name)
							#print("energy_match ", energy_match, graph_name)
							if energy_match:
								try:
									talys_energy_100keV = int(energy_match.group(1))
									talys_energy_keV = talys_energy_100keV / 100.0
									
									diff = abs(talys_energy_keV - energy_keV)
									#print("energy: ", talys_energy_keV, energy_keV, diff, min_diff)
									if diff < min_diff and diff < tolerance:
										min_diff = diff
										best_graph = str(graph_name)
										best_talys_energy = talys_energy_keV
								except ValueError:
									continue
						#print(best_graph)
						if best_graph:
							graph_info = match.copy()
							graph_info['graph_name'] = best_graph
							graph_info['found_energy'] = best_talys_energy
							graph_info['graph_energy_diff'] = min_diff
							talys_graphs_info.append(graph_info)
							print(f"Found TALYS graph for {isotope}: {best_graph} (Eγ={best_talys_energy:.1f} keV)")
							
				except Exception as e:
					print(f"Error reading TALYS ED file for {isotope}: {e}")
			else:
				print(f"TALYS file not found: {talys_ed_file}")
		
		return talys_graphs_info

	def calculate_talys_sum(self, talys_graphs_info, external_data_dir, x_range):
		"""Вычисление суммы расчетов TALYS для близких энергий"""
		talys_sum = None
		sum_abundance = 1
		check_isotope = set()
		
		if len(talys_graphs_info)<2:
			return talys_sum
			
		for talys_info in talys_graphs_info:
			isotope = talys_info['mother_nucleus']
			graph_name = talys_info['graph_name']
			abundance = talys_info['abundance']
			
			talys_ed_file = os.path.join(external_data_dir, "TalysED", "root", f"{isotope}.root")

		  
			if os.path.exists(talys_ed_file):
				try:
					x_talys, y_talys, _, _ = self.get_tgraph_data(talys_ed_file, graph_name)
					if x_talys is not None and y_talys is not None:
						# Интерполируем на общую сетку x_range
						y_interp = np.interp(x_range, x_talys, y_talys)
						
						# Умножаем на abundance (выход гамма-квантов)
						weighted_y = y_interp * abundance
						
						if talys_sum is None:
							talys_sum = weighted_y
							if(isotope not in check_isotope):
								sum_abundance = abundance
								check_isotope.add(isotope)
						else:
							talys_sum += weighted_y
							if(isotope not in check_isotope):
								sum_abundance += abundance
								check_isotope.add(isotope)
							
				except Exception as e:
					print(f"Error calculating TALYS sum for {isotope}: {e}")
		
		return talys_sum / sum_abundance

	def extract_energy_from_graph_name(self, graph_name):
		"""Извлечение энергии из имени TGraph"""
		# Преобразуем в строку на случай если это pandas объект
		graph_name_str = str(graph_name)
		pattern = r'ADistD_(\d+)_LaBr'
		match = re.search(pattern, graph_name_str)
		if match:
			try:
				energy_102eV = int(match.group(1))
				return energy_102eV / 10.0  # Преобразование в кэВ
			except ValueError:
				pass
		return None
		
	#%%add_to NuclearDataComparator
	def get_tgraph_data(self, root_file, graph_name):
		"""Извлечение данных из TGraphErrors"""
		try:
			with uproot.open(root_file) as file:
				graph = file[graph_name]
				
				x_values = graph.member('fX')
				y_values = graph.member('fY')
				
				# Пробуем получить ошибки
				try:
					x_errors = graph.member('fEX')
				except:
					x_errors = np.zeros_like(x_values)
				
				try:
					y_errors = graph.member('fEY')
				except:
					y_errors = np.zeros_like(y_values)
				
				return x_values, y_values, x_errors, y_errors
				
		except Exception as e:
			print(f"Ошибка при чтении TGraph {graph_name}: {e}")
			return None, None, None, None

	def get_sorted_graphs_by_energy(self, root_file):
		"""Извлечение и сортировка графиков по энергии гамма-квантов"""
		sorted_graphs = []
		
		try:
			with uproot.open(root_file) as file:
				for graph_name in file.keys():
					graph_name_str = str(graph_name)
					
					# Пропускаем не TGraphErrors объекты
					if 'ADistD_' not in graph_name_str or 'LaBr' not in graph_name_str:
						continue
					
					# Извлекаем энергию из имени графика
					energy_keV = self.extract_energy_from_graph_name(graph_name_str)
					if energy_keV is None:
						continue
					
					sorted_graphs.append({
						'name': graph_name_str,
						'energy_keV': energy_keV
					})
			
			# Сортируем графики по энергии (по возрастанию)
			sorted_graphs.sort(key=lambda x: x['energy_keV'])
			
			print(f"Found {len(sorted_graphs)} graphs, sorted by energy:")
			for i, graph_info in enumerate(sorted_graphs):
				print(f"  {i+1}: {graph_info['name']} - {graph_info['energy_keV']:.1f} keV")
			
			return sorted_graphs
			
		except Exception as e:
			print(f"Error reading ROOT file: {e}")
			return []

	def parse_external_filename(self, filename, data_type="AD"):
		"""Парсинг имени файла внешних данных"""
		# Преобразуем в строку на случай если это не строка
		filename_str = str(filename)
		clean_name = filename_str.replace('.txt', '')
		
		if data_type == "AD":  # Angular Distribution
			pattern = r'(\d+[A-Za-z]+)_EG_([\d.]+)_En_([\d.]+)_([A-Za-z_]+)_(.+)'
		elif data_type == "ED":  # ED - Energy Distribution
			pattern = r'(\d+[A-Za-z]+)_EG_([\d.]+)_([A-Za-z_]+)_(.+)'
		
		match = re.match(pattern, clean_name)
		
		if match:
			try:
				if data_type == "AD":
					return {
						'isotope': match.group(1),
						'gamma_energy': float(match.group(2)),
						'neutron_energy': float(match.group(3)),
						'data_type': match.group(4),
						'author': match.group(5)
					}
				elif data_type == "ED":
					return {
						'isotope': match.group(1),
						'gamma_energy': float(match.group(2)),
						'data_type': match.group(3),
						'author': match.group(4)
					}
			except ValueError:
				pass
		return None

	#%%add_to NuclearDataComparator
	
	# not used
	def find_matching_external_files(self, base_directory, isotope, gamma_energy, data_type="AD", tolerance=10):
		"""Поиск файлов, соответствующих изотопу и энергии"""
		if data_type == "AD":
			isotope_dir = os.path.join(base_directory, "GammaAD", "dat", isotope)
		else:
			isotope_dir = os.path.join(base_directory, "GammaED", "dat", isotope)
		
		if not os.path.exists(isotope_dir):
			print(f"Директория не найдена: {isotope_dir}")
			return []
		
		try:
			all_files = os.listdir(isotope_dir)
			txt_files = [f for f in all_files if f.endswith('.txt') ]
			#print("txt ", txt_files)
		except Exception as e:
			print(f"Ошибка при чтении директории {isotope_dir}: {e}")
			return []
		
		# Группируем файлы по авторам для AD данных
		author_files = {} if data_type == "AD" else None
		matching_files = []
		
		for filename in txt_files:
			file_info = self.parse_external_filename(filename, data_type)
			if file_info:
				#print("energy ", file_info['gamma_energy'], gamma_energy)
				gamma_match = abs(file_info['gamma_energy'] - gamma_energy) < tolerance
				
				if data_type == "AD":
					neutron_match = abs(file_info['neutron_energy'] - self.target_neutron_energy) <= self.neutron_energy_tolerance
					if gamma_match and neutron_match:
						full_path = os.path.join(isotope_dir, filename)
						author = file_info['author']
						
						if author not in author_files:
							author_files[author] = []
						
						author_files[author].append({
							'path': full_path,
							'neutron_energy': file_info['neutron_energy'],
							'file_info': file_info
						})
				elif data_type == "ED":
					if gamma_match:
						full_path = os.path.join(isotope_dir, filename)
						matching_files.append(full_path)
		
		# Для AD данных выбираем по одному файлу на автора с ближайшей энергией нейтрона
		if data_type == "AD" and author_files:
			for author, files in author_files.items():
				if len(files) > 1:
					# Сортируем по близости к целевой энергии нейтрона
					files.sort(key=lambda x: abs(x['neutron_energy'] - self.target_neutron_energy))
					best_file = files[0]
					print(f"Автор {author}: выбрана энергия нейтрона {best_file['neutron_energy']} МэВ "
						  f"(вместо {[f['neutron_energy'] for f in files]})")
					matching_files.append(best_file['path'])
				else:
					matching_files.append(files[0]['path'])
		
		return matching_files

	#%%add_to NuclearDataComparator
	def load_external_data(self, filepath, data_type="AD"):
		"""Загрузка данных из внешнего файла"""
		metadata = {}
		data = pd.DataFrame()
		
		try:
			with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
				lines = f.readlines()
			
			# Парсим заголовок
			header_lines = [line for line in lines if '#' in line]
			for line in header_lines:
				if '_' in line:
					parts = line.strip('#').strip().split('_')
					for i in range(0, len(parts)-1, 2):
						if i+1 < len(parts):
							metadata[parts[i]] = parts[i+1]
			
			# Загрузка данных
			data_lines = [line for line in lines if not '#' in line and line.strip()]
			if data_lines:
				if data_type == "AD":
					data = pd.read_csv(filepath, comment='#', sep=r"\s+",
								header=None, names=['angle', 'angle_err', 'cs', 'cs_err'])
					# Преобразуем углы в косинусы
					data['cos_angle'] = np.cos(np.radians(data['angle']))
					data['cos_err'] = np.abs(np.sin(np.radians(data['angle'])) * np.radians(data['angle_err']))
				else:
					data = pd.read_csv(filepath, comment='#', sep=r'\s+',
									  header=None, names=['energy', 'energy_err', 'cs', 'cs_err'])
				
		except Exception as e:
			print(f"Ошибка при чтении файла {filepath}: {e}")
		
		return metadata, data

	#%%add_to NuclearDataComparator

	def find_external_files_for_all_isotopes(self, external_data_dir, all_talys_matches, energy_keV, data_type="AD", tolerance = 5.0):
		"""Поиск внешних файлов для всех изотопов с близкими энергиями без дубликатов"""
		all_external_files = []
		seen_authors = set()  # Для отслеживания уникальных авторов
		
		for talys_match in all_talys_matches:
			isotope = talys_match['mother_nucleus']
			talys_energy = talys_match['energy_keV']
			
			# Ищем файлы для текущего изотопа
			if data_type == "AD":
				isotope_dir = os.path.join(external_data_dir, "GammaAD", "dat", isotope)
			else:
				isotope_dir = os.path.join(external_data_dir, "GammaED", "dat", isotope)
			
			if not os.path.exists(isotope_dir):
				continue
				
			try:
				all_files = os.listdir(isotope_dir)
				txt_files = [f for f in all_files if f.endswith('.txt')]
			except Exception as e:
				print(f"Ошибка при чтении директории {isotope_dir}: {e}")
				continue
			
			for filename in txt_files:
				file_info = self.parse_external_filename(filename, data_type)
				if not file_info:
					continue
					
				# Создаем уникальный ключ автора (автор + год + тип данных)
				author_label = self.get_author_label({}, file_info)  # Получаем автор+год
				author_key = f"{author_label}_{data_type}"
				
				# Проверяем соответствие энергии (с допуском)
				gamma_match = abs(file_info['gamma_energy'] - energy_keV) < tolerance
				
				if data_type == "AD":
					neutron_match = abs(file_info['neutron_energy'] - self.target_neutron_energy) <= self.neutron_energy_tolerance
					if not (gamma_match and neutron_match):
						continue
				elif data_type == "ED":
					if not gamma_match:
						continue
				
				full_path = os.path.join(isotope_dir, filename)
				energy_diff = abs(file_info['gamma_energy'] - energy_keV)
				
				# Если автор уже встречался, проверяем, не нашли ли мы более близкий файл
				if author_key in seen_authors:
					# Ищем существующий файл этого автора
					for existing_file in all_external_files:
						if existing_file['author_key'] == author_key:
							# Если нашли более близкий файл, заменяем
							if energy_diff < existing_file['energy_diff']:
								print(f"Replacing {existing_file['path']} with {full_path} "
									  f"(better energy match: {energy_diff:.2f} vs {existing_file['energy_diff']:.2f} keV)")
								all_external_files.remove(existing_file)
								all_external_files.append({
									'path': full_path,
									'file_info': file_info,
									'isotope': isotope,
									'talys_energy': talys_energy,
									'energy_diff': energy_diff,
									'author_key': author_key
								})
							break
				else:
					# Новый автор - добавляем
					seen_authors.add(author_key)
					all_external_files.append({
						'path': full_path,
						'file_info': file_info,
						'isotope': isotope,
						'talys_energy': talys_energy,
						'energy_diff': energy_diff,
						'author_key': author_key
					})
		
		# Сортируем по разнице энергий (самые близкие сначала)
		all_external_files.sort(key=lambda x: x['energy_diff'])
		return all_external_files

	def get_unique_external_files(self, external_data_dir, all_talys_matches, energy_keV, data_type="AD", tolerance = 5.0):
		"""Получение уникальных внешних файлов для всех изотопов"""
		# Собираем все файлы для всех изотопов (уже без дубликатов)
		all_files = self.find_external_files_for_all_isotopes(external_data_dir, all_talys_matches, energy_keV, data_type, tolerance)
		
		print(f"Found {len(all_files)} {data_type} files for all isotopes:")
		for file_data in all_files:
			author_label = self.get_author_label({}, file_data['file_info'])
			print(f"  {file_data['isotope']}: {author_label} "
				  f"(Eγ={file_data['file_info']['gamma_energy']:.1f} keV, diff={file_data['energy_diff']:.2f} keV)")
		
		# Просто возвращаем пути к файлам
		return [file_data['path'] for file_data in all_files]
		

	def get_correct_error_fit(self, fit_function, parameter_index):
		"""Расчет корректной ошибки параметра фита"""
		ndf = fit_function.GetNDF()
		chi2 = fit_function.GetChisquare()
		
		if ndf != 0 and chi2 != 0:
			xi_sqrt = np.sqrt(chi2 / ndf)
			return fit_function.GetParError(parameter_index) * xi_sqrt
		return fit_function.GetParError(parameter_index)

	#%%add_to NuclearDataComparator
	def fit_legendre_polynomials(self, angles_deg, values, errors, multipolarity):
		"""Фит полиномами Лежандра с корректными ошибками"""
		# Преобразуем углы в косинусы
		cos_theta = np.cos(np.radians(angles_deg))
		
		# Создаем TGraphErrors для фита в ROOT
		n_points = len(angles_deg)
		graph = ROOT.TGraphErrors(n_points)
		
		for i in range(n_points):
			graph.SetPoint(i, cos_theta[i], values[i])
			graph.SetPointError(i, 0, errors[i])
		
		# Создаем функцию для фита
		n_params = multipolarity + 2  # E1: 2 параметра, E2: 3 параметра
		fit_function = ROOT.TF1("fit_func", self._legendre_polynomial_sum, -1, 1, n_params)
		
		# Передаем мультипольность как параметр
		fit_function.SetParameter(0, multipolarity)
		fit_function.FixParameter(0, multipolarity)
		
		# Устанавливаем начальные параметры для коэффициентов
		for i in range(1, n_params):
			fit_function.SetParName(i, f"a{2*(i-1)}")
			fit_function.SetParameter(i, 1.0)
		
		# Выполняем фит
		graph.Fit(fit_function, "Q")
		
		# Извлекаем параметры с корректными ошибками
		params = [fit_function.GetParameter(i) for i in range(1, n_params)]
		param_errors = [self.get_correct_error_fit(fit_function, i) for i in range(1, n_params)]
		
		# Полное сечение = a0 * 4π
		total_cs = params[0] * 4 * np.pi
		total_cs_error = param_errors[0] * 4 * np.pi
		
		# Создаем фитированную кривую
		fit_x = np.linspace(-1, 1, 100)
		fit_y = [fit_function.Eval(x) for x in fit_x]
		
		return fit_x, fit_y, total_cs, total_cs_error, params, param_errors, fit_function.GetChisquare(), fit_function.GetNDF()

	#%%add_to NuclearDataComparator
	def _legendre_polynomial_sum(self, x, par):
		"""Сумма полиномов Лежандра для фита с использованием sympy"""
		xx = x[0]
		result = 0.0
		
		# Первый параметр - мультипольность
		multipolarity = int(par[0])
		
		# Суммируем четные полиномы Лежандра до 2*multipolarity
		for i in range(multipolarity + 1):
			degree = 2 * i
			coefficient = par[i + 1]
			
			# Вычисляем значение полинома Лежандра
			if degree == 0:
				legendre_value = 1.0
			elif degree == 2:
				legendre_value = 0.5 * (3 * xx**2 - 1)
			elif degree == 4:
				legendre_value = 0.125 * (35 * xx**4 - 30 * xx**2 + 3)
			elif degree == 6:
				legendre_value = 0.0625 * (231 * xx**6 - 315 * xx**4 + 105 * xx**2 - 5)
			else:
				# Для других степеней используем sympy
				t = sp.symbols('t')
				legendre_poly = sp.legendre(degree, t)
				legendre_value = float(legendre_poly.subs(t, xx))
		
			result += coefficient * legendre_value
		return result

	def find_max_multipolarity(self, all_talys_matches, energy_keV, tolerance=10):
		"""Нахождение максимальной мультипольности среди всех подходящих энергий TALYS"""
		max_multipolarity = 0
		best_match = None
		
		for match in all_talys_matches:
			# Проверяем близость энергии
			energy_diff = abs(match['energy_keV'] - energy_keV)
			if energy_diff < tolerance:
				if match['multipolarity'] > max_multipolarity:
					max_multipolarity = match['multipolarity']
					best_match = match
		
		if best_match is not None:
			print(f"Selected multipolarity {max_multipolarity} from {best_match['mother_nucleus']} "
				  f"({best_match['energy_keV']:.1f} keV, {best_match['reaction']})")
		else:
			print(f"No suitable multipolarity found for energy {energy_keV:.1f} keV")
			# Возвращаем первую попавшуюся мультипольность или значение по умолчанию
			if all_talys_matches:
				max_multipolarity = all_talys_matches[0]['multipolarity']
				print(f"Using default multipolarity {max_multipolarity}")
		
		return max_multipolarity, best_match
		
	def get_author_style(self, author_name):
		"""Получение или создание стиля для автора"""
		if author_name not in self.author_styles:
			# Назначаем новый стиль автору
			marker = self.markers[self.marker_index % len(self.markers)]
			color = self.colors[self.color_index % len(self.colors)]
			
			self.author_styles[author_name] = {
				'marker': marker,
				'color': color
			}
			
			# Обновляем индексы для следующих авторов
			self.marker_index += 1
			self.color_index += 1
			
			# Если закончились маркеры или цвета, начинаем сначала но с другим сочетанием
			if self.marker_index >= len(self.markers):
				self.marker_index = 0
			if self.color_index >= len(self.colors):
				self.color_index = 0
		
		return self.author_styles[author_name]
		
	#%%add_to NuclearDataComparator
	def get_author_label(self, metadata, file_info):
		"""Создание подписи для легенды - только фамилия и год"""
		author = file_info.get('author', 'Unknown')
		
		# Извлекаем только фамилию (последнее слово)
		# Убираем все точки, запятые, цифры и символы
		clean_author = re.sub(r'[.,\d_]', ' ', author)
		
		# Разделяем по пробелам и берем последнее слово как фамилию
		parts = clean_author.split()
		if parts:
			surname = parts[-1]
		else:
			# Если нет пробелов, пытаемся разделить по заглавным буквам
			# Для формата типа "RONelson" -> "Nelson"
			matches = re.findall(r'[A-Z][a-z]+', clean_author)
			surname = matches[-1] if matches else clean_author
		
		# Извлекаем год
		year_match = re.search(r'(\d{4})', author)
		year = year_match.group(1) if year_match else metadata.get('Year', '')
		
		if year:
			return f"{surname}{year}"
		return surname
		
	def print_author_styles(self):
		"""Вывод информации о назначенных стилях авторов"""
		print("\n=== НАЗНАЧЕННЫЕ СТИЛИ АВТОРОВ ===")
		for author, style in self.author_styles.items():
			print(f"{author}: маркер={style['marker']}, цвет={style['color']}")
			
	def extract_elements_from_filename(self, filename):
		"""Извлечение всех химических элементов из имени файла"""
		# Регулярное выражение для поиска химических элементов
		# Заглавная буква, за которой может следовать строчная
		pattern = r'[A-Z][a-z]?'
		elements = re.findall(pattern, filename)
		return elements

	def select_sample_file(self, element, root_file_dir):
		"""Выбор файла образца с точным поиском химических элементов"""
		if not os.path.exists(root_file_dir):
			print(f"Directory not found: {root_file_dir}")
			return None
		
		try:
			root_files = [f for f in os.listdir(root_file_dir) if f.endswith('.root')]
		except Exception as e:
			print(f"Error reading directory {root_file_dir}: {e}")
			return None
		
		# Точный поиск по химическому элементу
		element_files = []
		
		for file in root_files:
			# Извлекаем все химические элементы из имени файла
			elements_in_file = self.extract_elements_from_filename(file)
			
			# Проверяем, содержится ли искомый элемент в списке
			if element in elements_in_file:
				element_files.append(file)
		
		if not element_files:
			print(f"No files found containing element {element} in {root_file_dir}")
			print(f"Available files: {root_files}")
			return None
		
		print(f"\nAvailable sample files containing {element}:")
		for i, file in enumerate(element_files):
			print(f"{i}: {file}")
		
		try:
			file_index = int(input("Select file number: "))
			if 0 <= file_index < len(element_files):
				selected_file = element_files[file_index]
				return os.path.join(root_file_dir, selected_file)
			else:
				print("Invalid selection")
				return None
		except ValueError:
			print("Please enter a valid number")
			return None

	def get_talys_file_path(self, element, talys_base_dir):
		"""Получение пути к файлу TALYS с точным поиском химических элементов"""
		if not os.path.exists(talys_base_dir):
			print(f"TALYS directory not found: {talys_base_dir}")
			return None
		
		# Точный поиск по имени элемента в химических формулах
		try:
			all_files = [f for f in os.listdir(talys_base_dir) if f.endswith('.txt')]
			for file in all_files:
				# Извлекаем все химические элементы из имени файла
				elements_in_file = self.extract_elements_from_filename(file)
				
				# Проверяем, содержится ли искомый элемент в списке
				if element in elements_in_file:
					talys_path = os.path.join(talys_base_dir, file)
					print(f"Found TALYS file: {talys_path}")
					return talys_path
		except Exception as e:
			print(f"Error reading TALYS directory: {e}")
		
		print(f"No TALYS file found for element {element} in {talys_base_dir}")
		return None

	def create_comparison_plots(self, angles_deg, values, angle_errors, value_errors,
							   fit_x, fit_y, total_cs, total_cs_error, fit_params, fit_errors,
							   chi2, ndf, external_ad_files, external_ed_files, 
							   isotope, energy_keV, multipolarity, reaction_info,
<<<<<<< HEAD
							   external_data_dir, output_pdf, graph_name, all_talys_matches, talys_graphs_info):
=======
							   external_data_dir, output_pdf, output_dir, graph_name, all_talys_matches, talys_graphs_info):
>>>>>>> d1da805 (Initial commit: Nuclear Data Comparator)
		"""Создание графиков с учетом ВСЕХ близких энергий TALYS и их суммы"""
		
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
		legend_handles = []
		legend_labels = []
		added_authors = set()
		
		# Преобразуем углы в косинусы
		cos_angles = np.cos(np.radians(angles_deg))
		cos_errors = np.abs(np.sin(np.radians(angles_deg)) * np.radians(angle_errors))
		
		# График 1: Angular distributions
		our_data = ax1.errorbar(cos_angles, values, xerr=cos_errors, yerr=value_errors,
							   fmt='o', markersize=11, capsize=0, color='black',
							   alpha=1, linewidth=2, elinewidth=1)
		legend_handles.append(our_data[0])
		legend_labels.append('Our data')
		added_authors.add('Our data')
		
		fit_line, = ax1.plot(fit_x, fit_y, 'r-', linewidth=2)
		legend_handles.append(fit_line)
		legend_labels.append(f'Fit (M={multipolarity})')
		added_authors.add(f'Fit (M={multipolarity})')
		
		# Данные других авторов
		for filepath in external_ad_files:
			try:
				filename = os.path.basename(filepath)
				file_info = self.parse_external_filename(filename, "AD")
				
				if file_info:
					metadata, data = self.load_external_data(filepath, "AD")
					
					if not data.empty:
						label = self.get_author_label(metadata, file_info)
						
						if label not in added_authors:
							style = self.get_author_style(file_info['author'])
							
							author_data = ax1.errorbar(data['cos_angle'], data['cs'], 
													  xerr=data['cos_err'], yerr=data['cs_err'],
													  fmt=style['marker'], markersize=9, capsize=0,
													  color=style['color'], alpha=0.8,
													  markeredgecolor='black', markeredgewidth=0.02,
													  elinewidth=1)
							
							legend_handles.append(author_data[0])
							legend_labels.append(label)
							added_authors.add(label)
						
			except Exception as e:
				print(f"Error processing AD file {filepath}: {e}")
		
		ax1.set_xlabel('cos(θ)')
		ax1.set_ylabel('dσ/dΩ, mb/sr')
		

		title_lines = [f"{reaction_info}, Eγ = {energy_keV:.1f} keV, M={multipolarity}"]
		
		for talys_data in all_talys_matches:
			talys_energy = talys_data['energy_keV']
			talys_cs = talys_data['cross_section']
			talys_abundance = talys_data['abundance']
			talys_reaction = talys_data['reaction']
			weighted_cs = talys_cs * talys_abundance
			
			if talys_abundance > 0:
				title_lines.append(f"{talys_energy:.1f} keV {talys_reaction}: {talys_cs:.1f} mb ({weighted_cs:.2f} mb)")
			else:
				title_lines.append(f"{talys_energy:.1f} keV {talys_reaction}: {talys_cs:.1f} mb")
				
		ax1.set_title('\n'.join(title_lines), fontsize=16, pad=20)
		ax1.set_xlim(-1, 1)
		ax1.grid(True, alpha=0.3)
		
		# Добавляем параметры фита на график
		fit_text = f"σ = {total_cs:.2f} ± {total_cs_error:.2f} mb\n"
		for i, (param, error) in enumerate(zip(fit_params, fit_errors)):
			fit_text += f"a{2*i} = {param:.3f} ± {error:.3f}\n"
		fit_text += f"χ²/NDF = {chi2:.1f}/{ndf}"
		
		
		ax1.text(0.02, 0.98, fit_text, transform=ax1.transAxes, verticalalignment='top',
				 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=14)
		
		# График 2: Total cross sections
		our_cs = ax2.errorbar(self.target_neutron_energy, total_cs, 
							 yerr=total_cs_error,
							 fmt='o', markersize=11, capsize=0, color='black',
							 alpha=1, linewidth=2, elinewidth=1)

		# Данные других авторов (полные сечения)
		for filepath in external_ed_files:
			try:
				filename = os.path.basename(filepath)
				file_info = self.parse_external_filename(filename, "ED")
				
				if file_info:
					metadata, data = self.load_external_data(filepath, "ED")
					
					if not data.empty:
						neutron_energy_min = self.target_neutron_energy - self.neutron_energy_tolerance
						neutron_energy_max = self.target_neutron_energy + self.neutron_energy_tolerance
						
						filtered_data = data[
							(data['energy'] >= neutron_energy_min) & 
							(data['energy'] <= neutron_energy_max)
						]
						
						if not filtered_data.empty:
							label = self.get_author_label(metadata, file_info)
							style = self.get_author_style(file_info['author'])
							
							author_cs = ax2.errorbar(filtered_data['energy'], filtered_data['cs'], 
													xerr=filtered_data['energy_err'],
													yerr=filtered_data['cs_err'],
													fmt=style['marker'], markersize=9, capsize=0,
													color=style['color'], alpha=0.8,
													markeredgecolor='black', markeredgewidth=0.02,
													elinewidth=1)
							
							if label not in added_authors:
								legend_handles.append(author_cs[0])
								legend_labels.append(label)
								added_authors.add(label)
							
			except Exception as e:
				print(f"Error processing ED file {filepath}: {e}")
		
		# Подготовка данных для суммы TALYS
		x_range = np.linspace(11.5, 16.5, 100)  # Диапазон энергий нейтронов
		talys_sum = self.calculate_talys_sum(talys_graphs_info, external_data_dir, x_range)
		
		# Рисуем индивидуальные графики TALYS
		colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
		for i, talys_info in enumerate(talys_graphs_info):
			isotope = talys_info['mother_nucleus']
			talys_energy = talys_info['energy_keV']
			graph_name = talys_info['graph_name']
			abundance = talys_info['abundance']
			reaction = talys_info['reaction']
			
			talys_ed_file = os.path.join(external_data_dir, "TalysED", "root", f"{isotope}.root")
			
			if os.path.exists(talys_ed_file):
				try:
					x_talys, y_talys, _, _ = self.get_tgraph_data(talys_ed_file, graph_name)
					if x_talys is not None and y_talys is not None:
						# Умножаем на abundance
						y_weighted = y_talys # * abundance
						
						color_idx = i % len(colors)
						talys_line, = ax2.plot(x_talys, y_weighted, 
											  color=colors[color_idx], 
											  linestyle='--',
											  linewidth=1.5, 
											  alpha=0.7,
											  label=f"TALYS {reaction} {talys_energy:.1f} keV")
						
						if f"TALYS {isotope}{reaction}{talys_energy}" not in added_authors:
							legend_handles.append(talys_line)
							legend_labels.append(f"TALYS {isotope}{reaction}{talys_energy}")
							added_authors.add(f"TALYS {isotope}{reaction}{talys_energy}")
							
				except Exception as e:
					print(f"Error plotting TALYS for {isotope}: {e}")
		
		# Рисуем сумму TALYS
		if talys_sum is not None:
			sum_line, = ax2.plot(x_range, talys_sum, 
								color='blue', 
								linestyle='-',
								linewidth=3, 
								alpha=0.9,
								label='TALYS Sum')
			
			legend_handles.append(sum_line)
			legend_labels.append('TALYS Sum')
			added_authors.add('TALYS Sum')
		
		
		ax2.set_xlabel(r'$\mathdefault{E_n}$, MeV', fontsize=16)
		ax2.set_ylabel('σ, mb', fontsize=16)
		ax2.set_title(f"Total cross sections for {reaction_info}", fontsize=14, pad=20)
		ax2.grid(True, alpha=0.3)
		ax2.set_xlim(12, 16)
		
		# Легенда наверху
		fig.legend(legend_handles, legend_labels, loc='upper center', 
				   bbox_to_anchor=(0.5, 1), ncol=5, fontsize=12, frameon=False)
		
		plt.tight_layout(rect=[0, 0, 1, 0.88])

		# Сохраняем в PDF
		output_pdf.savefig(fig)
<<<<<<< HEAD
		plt.close()
		
		print(f"Plot created for {isotope}, Eγ = {energy_keV:.1f} keV")
		
=======
		
		svg_output_dir = os.path.join(output_dir, "svg", f"{self.selected_element}_{self.sample_name}")
		os.makedirs(svg_output_dir, exist_ok=True)
		
		svg_filename = f"{self.selected_element}_{self.sample_name}_Egamma_{energy_keV:.1f}keV.svg"
		svg_path = os.path.join(svg_output_dir, svg_filename)
		plt.savefig(svg_path, format='svg', bbox_inches='tight')
		print(f"SVG saved: {svg_path}")
    
		plt.close()
    
		print(f"Plot created for {isotope}, Eγ = {energy_keV:.1f} keV")
    
>>>>>>> d1da805 (Initial commit: Nuclear Data Comparator)
		return total_cs, total_cs_error
		

	def plot_comparison(self, root_file_dir, talys_base_dir, external_data_dir, output_dir="results"):
		"""Основная функция с сортировкой графиков по энергии"""
		
		if self.selected_element is None:
			element = input("Enter element to analyze (e.g., O, Ca): ").strip()
			if not self.set_element(element):
				return
		
		# Выбор файла образца
		root_file = self.select_sample_file(self.selected_element, root_file_dir)
		if root_file is None:
			return
		
		print(f"Selected ROOT file: {root_file}")
		
		# Поиск файла TALYS
		talys_file = self.get_talys_file_path(self.selected_element, talys_base_dir)
		if talys_file is None:
			return
		
		print(f"Selected TALYS file: {talys_file}")
		
		# Загружаем данные TALYS
		if self.parse_talys_file(talys_file) is None:
			return
		
		# Получаем отсортированные по энергии графики
		sorted_graphs = self.get_sorted_graphs_by_energy(root_file)
		if not sorted_graphs:
			print("No valid graphs found in ROOT file")
			return
		
		# Создаем PDF файл
		os.makedirs(output_dir, exist_ok=True)
		sample_name = os.path.basename(root_file).replace('.root', '').replace('_out', '')
<<<<<<< HEAD
=======
		self.sample_name = sample_name
>>>>>>> d1da805 (Initial commit: Nuclear Data Comparator)
		pdf_filename = f"{self.selected_element}_{sample_name}_analysis.pdf"
		pdf_path = os.path.join(output_dir, pdf_filename)
		
		with matplotlib.backends.backend_pdf.PdfPages(pdf_path) as pdf:
			processed_count = 0
			total_cs_results = []
			
			# Обрабатываем графики в порядке возрастания энергии
			for graph_info in sorted_graphs:
				graph_name_str = graph_info['name']
				energy_keV = graph_info['energy_keV']
				
				print(f"\n{'='*50}")
				print(f"Processing graph ({processed_count + 1}/{len(sorted_graphs)}): {graph_name_str}")
				print(f"Energy: {energy_keV:.1f} keV")
				print(f"{'='*50}")
				
				# Проверяем, есть ли энергия в данных TALYS
				if not self.is_energy_in_talys(energy_keV, tolerance=gamma_tolerance):
					print(f"Energy {energy_keV:.1f} keV not found in TALYS data")
					continue
				
				# Находим ВСЕ близкие энергии TALYS для всех изотопов
				all_talys_matches = self.find_all_talys_matches(energy_keV, tolerance=gamma_tolerance)
				
				if not all_talys_matches:
					print(f"No TALYS matches found for energy {energy_keV:.1f} keV")
					continue
				
				print(f"Found {len(all_talys_matches)} TALYS matches:")
				for match in all_talys_matches:
					print(f"  {match['mother_nucleus']}: {match['energy_keV']:.1f} keV "
						  f"({match['reaction']}, M={match['multipolarity']}, abundance={match['abundance']:.3f})")
				
				# Выбираем максимальную мультипольность
				multipolarity, best_talys_match = self.find_max_multipolarity(all_talys_matches, energy_keV, tolerance = gamma_tolerance)
				
				# Используем лучший матч для основной информации
				if best_talys_match:
					isotope = best_talys_match['mother_nucleus']
					reaction = best_talys_match['reaction']
					final_nucleus = best_talys_match['final_nucleus']
				else:
					isotope = all_talys_matches[0]['mother_nucleus']
					reaction = all_talys_matches[0]['reaction']
					final_nucleus = all_talys_matches[0]['final_nucleus']
				
				print(f"Selected: {isotope}, Eγ={energy_keV:.1f} keV, M={multipolarity}")
				
				# Ищем внешние файлы данных ДЛЯ ВСЕХ ИЗОТОПОВ
				external_ad_files = self.get_unique_external_files(
					external_data_dir, all_talys_matches, energy_keV, "AD", tolerance = gamma_tolerance
				)
				
				external_ed_files = self.get_unique_external_files(
					external_data_dir, all_talys_matches, energy_keV, "ED", tolerance = gamma_tolerance
				)
				
				print(f"Found {len(external_ad_files)} unique AD files, {len(external_ed_files)} unique ED files")
				
				# Находим соответствующие графики TALYS
				talys_graphs_info = self.find_talys_graphs_for_all_matches(
					all_talys_matches, external_data_dir, tolerance=gamma_tolerance
				)
				
				if not talys_graphs_info:
					print("No TALYS graphs found for the matches")
					continue
				
				print(f"Found {len(talys_graphs_info)} TALYS graphs")
				
				# Создаем информацию о реакции с учетом всех близких пиков
				reaction_info_parts = []
				for talys_data in all_talys_matches:
					reaction_str = f"{talys_data['mother_nucleus']} {talys_data['reaction']} {talys_data['final_nucleus']}"
					if talys_data['abundance'] > 0:
						reaction_str += f" ({talys_data['abundance']*100:.1f}%)"
					reaction_info_parts.append(reaction_str)
				
				reaction_info = " + ".join(reaction_info_parts)
				
				# Извлекаем данные из TGraphErrors
				angles_deg, values, angle_errors, value_errors = self.get_tgraph_data(
					root_file, graph_name_str
				)
				
				if angles_deg is None:
					print(f"Could not extract data from {graph_name_str}")
					continue
				
				print(f"Extracted {len(angles_deg)} data points")
				
				# Выполняем фит полиномами Лежандра с выбранной мультипольностью
				print(f"Performing Legendre polynomial fit with M={multipolarity}...")
				fit_result = self.fit_legendre_polynomials(
					angles_deg, values, value_errors, multipolarity
				)
				
				if fit_result is None:
					print("Fit failed")
					continue
				
				fit_x, fit_y, total_cs, total_cs_error, fit_params, fit_errors, chi2, ndf = fit_result
				
				print(f"Fit successful: σ = {total_cs:.2f} ± {total_cs_error:.2f} mb")
				print(f"χ²/NDF = {chi2:.1f}/{ndf}")
				print(f"Fit parameters: {fit_params}")
				
				# Создаем графики сравнения
				cs, cs_err = self.create_comparison_plots(
					angles_deg, values, angle_errors, value_errors,
					fit_x, fit_y, total_cs, total_cs_error, fit_params, fit_errors,
					chi2, ndf, external_ad_files, external_ed_files,
					isotope, energy_keV, multipolarity, reaction_info,
<<<<<<< HEAD
					external_data_dir, pdf, graph_name_str, all_talys_matches, talys_graphs_info
=======
					external_data_dir, pdf, output_dir, graph_name_str, all_talys_matches, talys_graphs_info
>>>>>>> d1da805 (Initial commit: Nuclear Data Comparator)
				)
				
				total_cs_results.append({
					'energy_keV': energy_keV,
					'total_cs': cs,
					'total_cs_error': cs_err,
					'isotope': isotope,
					'reaction': reaction_info,
					'multipolarity': multipolarity,
					'chi2': chi2,
					'ndf': ndf,
					'graph_name': graph_name_str,
					'selected_from': best_talys_match['mother_nucleus'] if best_talys_match else isotope,
					'num_ad_files': len(external_ad_files),
					'num_ed_files': len(external_ed_files)
				})
				
				processed_count += 1
				print(f"Completed processing graph {processed_count}/{len(sorted_graphs)}")
		
			
			# Выводим сводку результатов
			print(f"\n{'='*60}")
			print("SUMMARY OF RESULTS (sorted by energy)")
			print(f"{'='*60}")
			for result in total_cs_results:
				print(f"{result['isotope']}, Eγ={result['energy_keV']:.1f} keV: "
					  f"σ={result['total_cs']:.2f} ± {result['total_cs_error']:.2f} mb, "
					  f"M={result['multipolarity']}, χ²/NDF={result['chi2']:.1f}/{result['ndf']}")
			
			print(f"\nTotal graphs processed: {processed_count}/{len(sorted_graphs)}")
			print(f"Results saved to: {pdf_path}")
			
			# Сохраняем результаты в CSV файл с сортировкой по энергии
			if total_cs_results:
				csv_filename = f"{self.selected_element}_{sample_name}_results.csv"
				csv_path = os.path.join(output_dir, csv_filename)
				results_df = pd.DataFrame(total_cs_results)
				# Сортируем по энергии на случай если порядок изменился
				results_df = results_df.sort_values('energy_keV')
				results_df.to_csv(csv_path, index=False)
				print(f"Results table saved to: {csv_path}")
			
			return total_cs_results
