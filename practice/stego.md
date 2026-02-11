# **Инструкция как воспроизвести демонстрацию скрытого shell-скрипта в изображении**  

---

## 🔧 Требования к среде

Для воспроизведения вам понадобится:
- Виртуальная машина с **Linux**;
- Доступ к терминалу;
- Графическая среда для проверки открытия изображений;
- Установленные инструменты: `exiftool`, `steghide`, `binwalk`, `imagemagick`, `python3-pip`.

---

## Подготовка исходных файлов (общая для всех методов)

### Создаем рабочую папку

```bash
mkdir ~/stego-demo
cd ~/stego-demo
```

### 0.1. Получение тестового изображения

```bash
# Скачиваем стандартное JPEG-изображение
wget -O original.jpg https://upload.wikimedia.org/wikipedia/commons/e/e0/JPEG_example_JPG_RIP_025.jpg

# Для PNG (для LSB-метода)
convert original.jpg original.png
```

### 0.2. Создание shell-скрипта-полезной нагрузки

```bash
cat > payload.sh << 'EOF'
#!/bin/bash
echo "========================================"
echo "[*] Привет, мир! Скрытый shell-скрипт запущен."
echo "[*] Время: $(date)"
echo "========================================"

# Попытка открыть терминал для визуального подтверждения
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal --title="PoC: Скрытый скрипт" -- bash -c "echo УСПЕХ! Скрипт выполнен из изображения.; read"
elif command -v xterm &> /dev/null; then
    xterm -hold -e "echo УСПЕХ! Скрипт выполнен из изображения."
else
    echo "[!] GUI-терминал не найден, но скрипт отработал."
fi
EOF

chmod +x payload.sh
```

---

## Метод 1: EOF-append (конкатенация в конец файла)

Простой, совместимый, работает с любыми форматами.
###  Внедрение
```bash
cat original.jpg payload.sh > stego_eof.jpg
```

### Проверка изображения
```bash
file stego_eof.jpg          # → JPEG image data
xdg-open stego_eof.jpg      # → откроется как картинка
```

### Извлечение и выполнение
```bash
ORIG_SIZE=$(stat -c%s original.jpg)
dd if=stego_eof.jpg of=extracted.sh bs=1 skip=$ORIG_SIZE
chmod +x extracted.sh
./extracted.sh
```

### Плюсы
- Простота, универсальность.
- Изображение валидно.

### Минусы
- Легко обнаруживается по размеру.
- Данные видны в hex-редакторе.

---

## Метод 2: Внедрение в EXIF-комментарий (через `exiftool`)

Использует стандартные метаданные JPEG.
### Внедрение
```bash
exiftool -Comment="$(base64 -w0 payload.sh)" original.jpg -o stego_exif.jpg
```

###  Извлечение
```bash
exiftool -Comment stego_exif.jpg -b | base64 -d > extracted.sh
chmod +x extracted.sh
./extracted.sh
```

### Плюсы
- Данные внутри структуры JPEG.
- Не увеличивает размер "подозрительно".

### Минусы
- Удаляется при загрузке в соцсети/мессенджеры.
- Ограничение ~64 КБ.

---

## Метод 3: Steghide (DCT-стеганография с шифрованием)

Меняет наименее значимые биты в частотной области JPEG.
###  Установка
```bash
sudo apt install steghide -y
```

### Внедрение
```bash
steghide embed -cf original.jpg -ef payload.sh -p "secret123" -sf stego_steghide.jpg
```

###  Извлечение
```bash
steghide extract -sf stego_steghide.jpg -p "secret123"
chmod +x payload.sh
./payload.sh
```

###  Плюсы
- Высокая скрытность (данные не видны в hex).
- Поддержка AES-шифрования.

###  Минусы
- Неустойчив к пересжатию.
- Обнаруживается статистическим анализом (`stegdetect`).

---

## Метод 4: Внедрение в пользовательские XMP-теги

Использует расширяемые метаданные.
###  Внедрение
```bash
exiftool -xmp:HiddenScript="$(base64 -w0 payload.sh)" original.jpg -o stego_xmp.jpg
```

### Извлечение
```bash
exiftool -xmp:HiddenScript stego_xmp.jpg -b | base64 -d > extracted.sh
chmod +x extracted.sh
./extracted.sh
```

###  Плюсы
- Выглядит как легитимные метаданные.
- Поддерживается профессиональными редакторами.

###  Минусы
- Удаляется при обработке изображения.
- Требует знания структуры XMP.

---

## Метод 5: LSB-стеганография в PNG (через `stegano`)

Скрытие в наименее значимых битах пикселей PNG.
###  Установка
```bash
pip3 install stegano
```

###  Внедрение
```bash
stegano-lsb hide -i original.png -o stego_lsb.png -m "$(cat payload.sh)"
```

###  Извлечение
```bash
stegano-lsb reveal -i stego_lsb.png > extracted.sh
chmod +x extracted.sh
./extracted.sh
```

### Плюсы
- Визуально неотличимо от оригинала.
- Данные скрыты на уровне пикселей.

###  Минусы
- Работает **только с PNG/BMP**.
- Уничтожается при любом пересжатии.
- Требует Python.

---

## Метод 6: Polyglot PNG с автоматическим исполнением

Файл одновременно является **валидным PNG** и **исполняемым скриптом**.
###  Создание
```bash
{
  echo '#!/bin/bash'
  echo 'main() {'
  tail -n +2 payload.sh | sed 's/^/  /'
  echo '  exit 0'
  echo '}'
  echo 'main'
  echo ''
  cat original.png
} > autoexec.png

chmod +x autoexec.png
```

###  Проверка как изображения
```bash
xdg-open autoexec.png  # → откроется как картинка (в большинстве систем)
```

###  Автоматическое исполнение
```bash
./autoexec.png
```
→ Откроется терминал с сообщением **"УСПЕХ! Скрипт выполнен из изображения."**

###  Плюсы
- Двойная природа: и изображение, и скрипт.
- Нет ошибок при выполнении (благодаря `exit 0`).

###  Минусы
- Требует запуска как программы (`./file`).
- Не все просмотрщики принимают такой PNG.

---

## Метод 7: `.desktop`-файл для "двойного клика → выполнение"

Моделирует социальную инженерию: пользователь видит **иконку изображения**, но при двойном клике **выполняется код**.

###  Создание
```bash
cat > photo.jpg.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=Photo.jpg
Comment=Double-click to view your photo
Exec=gnome-terminal --title="PoC: Стеганография" -- bash -c "echo '📌 ВАЖНО: вы запустили скрипт двойным кликом!'; read"
Icon=image-jpeg
Terminal=false
StartupNotify=false
Categories=Graphics;
EOF

chmod +x photo.jpg.desktop
```

###  Использование
1. В файловом менеджере найдите файл **`Photo.jpg`** (на самом деле — `photo.jpg.desktop`).
2. **Дважды кликните** по нему.
3. Выберите **«Запустить»** (если спросит).
4. Откроется терминал с подтверждающим сообщением.

---

## Как проверить, что всё работает?

Для любого файла `stego_*.jpg` или `stego_*.png` выполните:

```bash
# 1. Проверка типа файла
file stego_eof.jpg

# 2. Открытие как изображения
xdg-open stego_eof.jpg

# 3. Анализ через binwalk (ищет вложения)
binwalk stego_eof.jpg

# 4. Просмотр метаданных
exiftool stego_exif.jpg

# 5. Сравнение размеров
stat -c "Оригинал: %s байт" original.jpg
stat -c "Stego:    %s байт" stego_eof.jpg
```

##  Приложение: Автоматический генератор всех методов

 `build_all_stego.sh`:

```bash
#!/bin/bash
set -e

# Подготовка
wget -O original.jpg https://upload.wikimedia.org/wikipedia/commons/e/e0/JPEG_example_JPG_RIP_025.jpg
convert original.jpg original.png
pip3 install stegano &>/dev/null

cat > payload.sh << 'EOF'
#!/bin/bash
echo "[*] Привет, мир! Скрытый скрипт."
if command -v xterm &> /dev/null; then xterm -hold -e "echo УСПЕХ!"; fi
EOF
chmod +x payload.sh

# Метод 1: EOF
cat original.jpg payload.sh > stego_eof.jpg

# Метод 2: EXIF
exiftool -Comment="$(base64 -w0 payload.sh)" original.jpg -o stego_exif.jpg

# Метод 3: Steghide
steghide embed -cf original.jpg -ef payload.sh -p "secret" -sf stego_steghide.jpg </dev/null

# Метод 4: XMP
exiftool -xmp:Payload="$(base64 -w0 payload.sh)" original.jpg -o stego_xmp.jpg

# Метод 5: LSB
stegano-lsb hide -i original.png -o stego_lsb.png -m "$(cat payload.sh)"

# Метод 6: Polyglot PNG
{
  echo '#!/bin/bash'
  echo 'main() {'
  tail -n +2 payload.sh | sed 's/^/  /'
  echo '  exit 0'
  echo '}'
  echo 'main'
  echo ''
  cat original.png
} > autoexec.png
chmod +x autoexec.png

# Метод 7: .desktop
cat > photo.jpg.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=Photo.jpg
Comment=Double-click to view
Exec=gnome-terminal --title="PoC" -- bash -c "echo '📌 Двойной клик → выполнение!'; read"
Icon=image-jpeg
Terminal=false
StartupNotify=false
EOF
chmod +x photo.jpg.desktop

echo "[*] Все файлы созданы:"
ls -lh stego_*.jpg stego_*.png autoexec.png photo.jpg.desktop
```

Запуск:
```bash
chmod +x build_all_stego.sh
./build_all_stego.sh
```

---
