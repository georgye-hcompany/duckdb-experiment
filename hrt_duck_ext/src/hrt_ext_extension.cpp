#define DUCKDB_EXTENSION_MAIN

#include "hrt_ext_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include "duckdb/common/vector_operations/unary_executor.hpp"

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

inline void HrtExtScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "...........🦆 " + name.GetString());
	});
}

inline void HrtExtOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "HrtExt " + name.GetString() + ", my linked OpenSSL version is " +
		                                           OPENSSL_VERSION_TEXT);
	});
}

static void HrtAddOneFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input = args.data[0];
	UnaryExecutor::Execute<int64_t, int64_t>(input, result, args.size(), [](int64_t x) {
		return x + 1;
	});
}

static void LoadInternal(ExtensionLoader &loader) {
	// Register a scalar function
	auto hrt_ext_scalar_function =
	    ScalarFunction("hrt_ext", {LogicalType::VARCHAR}, LogicalType::VARCHAR, HrtExtScalarFun);

	loader.RegisterFunction(hrt_ext_scalar_function);

	// Register another scalar function
	auto hrt_ext_openssl_version_scalar_function = ScalarFunction("hrt_ext_openssl_version", {LogicalType::VARCHAR},
	                                                             LogicalType::VARCHAR, HrtExtOpenSSLVersionScalarFun);
	loader.RegisterFunction(hrt_ext_openssl_version_scalar_function);

	auto hrt_add_one =
	    ScalarFunction("hrt_add_one", {LogicalType::BIGINT}, LogicalType::BIGINT, HrtAddOneFunction);
	loader.RegisterFunction(hrt_add_one);
}

void HrtExtExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string HrtExtExtension::Name() {
	return "hrt_ext";
}

std::string HrtExtExtension::Version() const {
#ifdef EXT_VERSION_HRT_EXT
	return EXT_VERSION_HRT_EXT;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(hrt_ext, loader) {
	duckdb::LoadInternal(loader);
}
}
