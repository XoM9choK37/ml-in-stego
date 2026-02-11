import subprocess
from PIL import Image

DATASET_SIZE = 10000

for i in range(1, DATASET_SIZE + 1):
    input_path = f"BOSSbase_1.01_256x256/{i}.pgm"
    temp_output_path = f"BOSSbase_1.01_bmp_256x256/{i}.bmp"
    steghide_output_path = f"SHELL_256x256_steghide_Anubis/{i}.bmp"
    stegano_output_path = f"SHELL_256x256_stegano_Anubis/{i}.bmp"
    
    image = Image.open(input_path)
    gray = image.convert('L')
    rgb = gray.convert("RGB")
    rgb.save(temp_output_path, format="BMP")
    
    steghide_command = f'steghide embed -cf {temp_output_path} -ef Anubis.sh -p "12345678" -sf {steghide_output_path}'
    stegano_command = f'stegano-lsb hide -i {temp_output_path} -o {stegano_output_path} -m "$(cat Anubis.sh)"'
    subprocess.run(steghide_command, shell=True, check=True)
    subprocess.run(stegano_command, shell=True, check=True)

for i in range(1, DATASET_SIZE + 1):
    input_path = f"BOSSbase_1.01_bmp_256x256/{i}.bmp"
    steghide_output_path = f"SHELL_256x256_steghide_AWFULSHRED/{i}.bmp"
    stegano_output_path = f"SHELL_256x256_stegano_AWFULSHRED/{i}.bmp"
    
    steghide_command = f'steghide embed -cf {input_path} -ef AWFULSHRED.sh -p "12345678" -sf {steghide_output_path}'
    stegano_command = f'stegano-lsb hide -i {input_path} -o {stegano_output_path} -m "$(cat AWFULSHRED.sh)"'
    subprocess.run(steghide_command, shell=True, check=True)
    subprocess.run(stegano_command, shell=True, check=True)

for i in range(1, DATASET_SIZE + 1):
    input_path = f"BOSSbase_1.01_bmp_256x256/{i}.bmp"
    steghide_output_path = f"SHELL_256x256_steghide_DarkRadiation/{i}.bmp"
    stegano_output_path = f"SHELL_256x256_stegano_DarkRadiation/{i}.bmp"
    
    steghide_command = f'steghide embed -cf {input_path} -ef DarkRadiation.sh -p "12345678" -sf {steghide_output_path}'
    stegano_command = f'stegano-lsb hide -i {input_path} -o {stegano_output_path} -m "$(cat DarkRadiation.sh)"'
    subprocess.run(steghide_command, shell=True, check=True)
    subprocess.run(stegano_command, shell=True, check=True)
    
for i in range(1, DATASET_SIZE + 1):
    input_path = f"BOSSbase_1.01_bmp_256x256/{i}.bmp"
    steghide_output_path = f"SHELL_256x256_steghide_IRCbot/{i}.bmp"
    stegano_output_path = f"SHELL_256x256_stegano_IRCbot/{i}.bmp"
    
    steghide_command = f'steghide embed -cf {input_path} -ef IRCbot.sh -p "12345678" -sf {steghide_output_path}'
    stegano_command = f'stegano-lsb hide -i {input_path} -o {stegano_output_path} -m "$(cat IRCbot.sh)"'
    subprocess.run(steghide_command, shell=True, check=True)
    subprocess.run(stegano_command, shell=True, check=True)
