# 输入文件的完整路径
input_file_path="$1"

# 获取输入文件的目录
input_directory=$(dirname "$input_file_path")

# 获取文件名，不带扩展名
input_filename=$(basename "$input_file_path" .jsonl)

# 计算原始jsonl文件的总行数
total_lines=$(wc -l < "$input_file_path")
echo "total lines: "$total_lines
# 要分成的文件数K，例如我们假设为4
K=$2

# 计算每个文件应该有多少行（向上取整）
lines_per_file=$(( (total_lines + K - 1) / K ))
echo "lines_per_file: "$lines_per_file
# 检查输出目录是否存在，如果不存在则创建它
mkdir -p "$input_directory"

# 使用split命令等分文件，并将结果文件放在输入文件的目录中，前缀为输入文件的名字
split -l $lines_per_file -d --additional-suffix=.jsonl "$input_file_path" "${input_directory}/${input_filename}-part-"