/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef XPROF_CONVERT_DATA_TABLE_UTILS_H_
#define XPROF_CONVERT_DATA_TABLE_UTILS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "nlohmann/json_fwd.hpp"  // from @com_github_nlohmann_json
#include "nlohmann/json.hpp"

namespace tensorflow {
namespace profiler {

static const char kBooleanTypeCode = 'B';
static const char kNumberTypeCode = 'N';
static const char kTextTypeCode = 'T';

class Value {
 public:
  explicit Value();
  Value(const Value&) = delete;
  Value& operator=(const Value&) = delete;
  virtual ~Value() = default;
  virtual const char& GetType() const = 0;
  virtual bool IsNull() const;

 protected:
  void set_null(bool null);

 private:
  bool null_;
};

class TextValue : public Value {
 public:
  TextValue();  // A NULL TextValue.
  explicit TextValue(absl::string_view value);
  TextValue(const TextValue&) = delete;
  TextValue& operator=(const TextValue&) = delete;
  ~TextValue() override = default;
  const char& GetType() const override;
  const std::string& GetValue() const;

 private:
  std::string value_;
};

class NumberValue : public Value {
 public:
  NumberValue();  // A NULL NumberValue.
  explicit NumberValue(double value);
  NumberValue(const NumberValue&) = delete;
  NumberValue& operator=(const NumberValue&) = delete;
  ~NumberValue() override = default;
  const char& GetType() const override;
  double GetValue() const;

 private:
  double value_;
};

class BooleanValue : public Value {
 public:
  BooleanValue();  // A NULL BooleanValue.
  explicit BooleanValue(bool value);
  BooleanValue(const BooleanValue&) = delete;
  BooleanValue& operator=(const BooleanValue&) = delete;
  ~BooleanValue() override = default;
  const char& GetType() const override;
  bool GetValue() const;

 private:
  bool value_;
};

struct TableCell {
  // Constructors with one argument - the value.
  explicit TableCell(bool value);
  explicit TableCell(double value);
  explicit TableCell(absl::string_view value);
  explicit TableCell(const char* value);

  // Constructors with two arguments - the value and the formatted value.
  TableCell(Value* value, absl::string_view formatted_value);
  TableCell(bool value, absl::string_view formatted_value);
  TableCell(double value, absl::string_view formatted_value);
  TableCell(absl::string_view value, absl::string_view formatted_value);
  TableCell(const char* value, absl::string_view formatted_value);

  // Constructors with two argument - value and custom properties.
  TableCell(Value* value,
            const absl::btree_map<std::string, std::string>& custom_properties);
  TableCell(bool value,
            const absl::btree_map<std::string, std::string>& custom_properties);
  TableCell(double value,
            const absl::btree_map<std::string, std::string>& custom_properties);
  TableCell(absl::string_view value,
            const absl::btree_map<std::string, std::string>& custom_properties);
  TableCell(const char* value,
            const absl::btree_map<std::string, std::string>& custom_properties);

  nlohmann::json GetCellValue() const;
  std::string GetCellValueStr() const;

  bool HasFormattedValue() const;

  const std::string& GetFormattedValue() const;

  std::unique_ptr<Value> value;
  std::unique_ptr<std::string> formatted_value;
  absl::btree_map<std::string, std::string> custom_properties;

  nlohmann::json GetCellValueImp(const Value* value) const;
  std::string GetCellValueStrImp(const Value* value) const;
};

struct TableColumn {
  TableColumn() = default;
  explicit TableColumn(std::string id, std::string type, std::string label)
      : id(id), type(type), label(label) {};
  explicit TableColumn(
      std::string id, std::string type, std::string label,
      absl::btree_map<std::string, std::string> custom_properties)
      : id(id), type(type), label(label), custom_properties(custom_properties) {
        };
  std::string id;
  std::string type;
  std::string label;
  absl::btree_map<std::string, std::string> custom_properties;
};

class TableRow {
 public:
  TableRow() = default;
  virtual ~TableRow() = default;

  TableRow& AddNumberCell(double value);
  TableRow& AddTextCell(absl::string_view value);
  TableRow& AddBooleanCell(bool value);
  TableRow& AddFormattedNumberCell(double value,
                                   absl::string_view formatted_value);
  TableRow& AddFormattedTextCell(absl::string_view value,
                                 absl::string_view formatted_value);
  TableRow& AddHexCell(uint64_t value);
  TableRow& AddBytesCell(uint64_t value);
  TableRow& AddFormattedDateCell(absl::Time value,
                                 absl::string_view value_format);

  std::vector<const TableCell*> GetCells() const;
  void SetCustomProperties(
      const absl::btree_map<std::string, std::string>& custom_properties);
  void AddCustomProperty(std::string name, std::string value);
  const absl::btree_map<std::string, std::string>& GetCustomProperties() const;
  int RowSize() const;

 private:
  std::vector<std::unique_ptr<TableCell>> cells_;
  absl::btree_map<std::string, std::string> custom_properties_;
};

class DataTable {
 public:
  DataTable() = default;

  void AddColumn(TableColumn column);
  const std::vector<TableColumn>& GetColumns() const;
  TableRow* AddRow();
  std::vector<const TableRow*> GetRows() const;
  void AddCustomProperty(std::string name, std::string value);
  std::string ToJson() const;

 private:
  std::vector<TableColumn> table_descriptions_;
  std::vector<std::unique_ptr<TableRow>> table_rows_;
  absl::btree_map<std::string, std::string> custom_properties_;
};

}  // namespace profiler
}  // namespace tensorflow
#endif  // XPROF_CONVERT_DATA_TABLE_UTILS_H_
