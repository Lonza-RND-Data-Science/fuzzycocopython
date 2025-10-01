/**
  * @file   logging.h
  * @author Karl Forner <karl.forner@gmail.com>
  * @date   09.2024
  * @section LICENSE
  * shamelessly borrowed from https://wandbox.org/permlink/i4P5cipoTsBqV2AL 
  * after reading this post: https://www.cppstories.com/2021/stream-logger/
  * 
  * 
*/
#ifndef LOGGING_LOGGER_H
#define LOGGING_LOGGER_H

#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <version>
#include <streambuf>
#include <memory>
#include <chrono>
#include <ctime>

namespace fuzzy_coco {

using namespace std;

#ifdef __cpp_lib_source_location
#include <source_location>
#define L_location location()
#else
#define L_location location(__FILE__,__LINE__, __func__)
#endif

namespace logging {
	enum Flags { L_clearFlags, L_concat, L_time, L_flush, L_endl, L_startWithFlushing, L_null, L_cout = 8, L_tabs = 16, L_allwaysFlush = 32 };
	inline Flags operator +=(Flags& l_flag, const Flags r_flag) { return l_flag = static_cast<Flags>(l_flag | r_flag); }
	inline Flags operator -=(Flags& l_flag, const Flags r_flag) { return l_flag = static_cast<Flags>(l_flag & ~r_flag); }

#ifdef __cpp_lib_source_location
	inline std::string location(const std::source_location& location = std::source_location::current()) {
		auto ss {std::stringstream {}};

		ss << location.file_name() << " : " << location.line() << " '" << location.function_name() << "'";
		return ss.str();
	}
#else
	inline std::string location(const char* file, int lineNo, const char* function) {
		auto ss = std::stringstream {};

		ss << file << " : " << lineNo << " '" << function << "'";
		return ss.str();
	}
#endif

	using Streamable = std::ostream;

	class Logger {
	public:
		void activate(bool makeActive = true) { makeActive ? _flags -= L_null : _flags += L_null; }
		Flags addFlag(Flags flag) { return _flags += flag; }
		Flags removeFlag(Flags flag) { return _flags -= flag; }
		virtual void flush() { stream().flush(); _flags -= L_startWithFlushing; }
		virtual bool open() { return false; }

		template<typename T>
		Logger& log(const T& value);
		Logger& operator <<(Flags);

		Logger& operator <<(decltype(std::endl<char, std::char_traits<char>>)) {
			return *this << L_endl;
		}

		Logger& operator <<(decltype(std::hex) manip) {
			stream() << manip;
			return *this;
		}

		Logger& operator <<(decltype(std::setw) manip) {
			stream() << manip;
			return *this;
		}

		virtual Streamable& stream();

		using ostreamPtr = Streamable*;
    virtual Logger* mirror_stream(ostreamPtr& mirrorStream)  = 0;
		// virtual Logger* mirror_stream(ostreamPtr& mirrorStream) {
		// 	mirrorStream = nullptr;
		// 	return this;
		// }

	protected:
		Logger(Flags initFlag = L_null) : _flags {initFlag} {}
		Logger(Flags initFlag = L_null, Streamable & = std::clog) : _flags {initFlag} {}

		virtual Logger& logTime();

		template<class T> friend Logger& operator <<(Logger& logger, const T& value);

		bool is_tabs() const { return _flags & L_tabs || has_time(); }
		bool is_null() const { return _flags == L_null; }
		bool is_cout() const { return _flags & L_cout; }
		bool has_time() const { return (_flags & 7) == L_time; }

		friend class FileNameGenerator;

		static tm* getTime();
		struct Log_date {
			unsigned char dayNo;
			unsigned char monthNo;
		} inline static log_date {0, 0};

		Flags _flags {L_startWithFlushing};
	};

	// Streaming template
	template<typename T>
	Logger& operator <<(Logger& logger, const T& value) {
		return logger.log(value);
	}

	template<typename T>
	Logger& Logger::log(const T& value) {
		if (is_null())
			return *this;

		auto streamPtr {&stream()};
		auto* logger {this};

		do {
			if (is_tabs()) {
				*streamPtr << "\t";
			}

			*streamPtr << value;
			logger = logger->mirror_stream(streamPtr);
		} while (streamPtr);

		removeFlag(L_time);
		return *this;
	}

	class Null_Buff : public std::streambuf { // derive because std::streambuf constructor is protected
	public:
		Null_Buff() { setp(nullptr, nullptr); }

	private:
		int_type overflow(int_type) override { return std::char_traits<char>::not_eof(0); }
	} inline null_buff {};

	inline Streamable null_ostream {&null_buff};

	inline Streamable& Logger::stream() { return null_ostream; }

	/// <summary>
	/// Logs to console - clog(default), cerr or cout.
	/// clog leaves flushing to the client (more efficient).
	/// cerr flushes on every operation, cout generally flushes at every new line.
	/// </summary>
	class Console_Logger : public Logger {
	public:
		Console_Logger(Flags initFlags = L_null, Streamable& ostream = std::clog)
			: Logger {initFlags, ostream}, _ostream {&ostream} {
			ostream.flush();
		}

		Streamable& stream() override { return is_null() ? Logger::stream() : *_ostream; }

		Logger* mirror_stream(ostreamPtr& mirrorStream) override {
			mirrorStream = mirrorStream == _ostream ? nullptr : _ostream;
			return this;
		}

	protected:
		Streamable* _ostream {nullptr};
	};

	Logger& logger(); // to be defined by the client

	// class FileNameGenerator {
	// public:
	// 	static constexpr int FILE_NAME_LENGTH {8};

	// 	FileNameGenerator(const std::filesystem::path& filePath);
	// 	std::string stem() const { return _fileNameStem; }
	// 	bool isNewDay(const Logger& logger) const { return _fileDayNo != logger.log_date.dayNo; }
	// 	int dayNo() const { return _fileDayNo; }
	// 	std::string operator()(const Logger& logger);

	// private:
	// 	std::string _fileNameStem;
	// 	std::filesystem::path _filePath;
	// 	unsigned char _fileDayNo {0};
	// };

	// /// <summary>: modified by Karl to remove the FileNameGenerator
	// /// Logs to file, and mirrors to the provided ostream - typcally clog
	// /// New Filenames are generated for each day
	// /// </summary>
	// template<typename MirrorBase = Console_Logger>
	// class SimpleFile_Logger : public MirrorBase {
	// public:
	// 	SimpleFile_Logger(const std::filesystem::path& filePath) : SimpleFile_Logger {filePath, L_null} {}
	// 	SimpleFile_Logger(const std::filesystem::path& filePath, Flags initFlags, Streamable& mirrorStream = std::clog);
	// 	SimpleFile_Logger(const std::filesystem::path& filePath, Flags initFlags, Logger& mirror_chain)
	// 	 : SimpleFile_Logger {filePath, initFlags} { _mirror = &mirror_chain; }

	// 	Streamable& stream() override;
	// 	void flush() override;
	// 	Logger* mirror_stream(Logger::ostreamPtr& mirrorStream) override;
	// 	bool open() override;

	// private:
	// 	Logger& logTime() override;
	// 	filesystem::path _file_path;
	// 	Logger* _mirror {this};
	// 	std::ofstream _dataFile;
	// };

	// template<typename MirrorBase>
	// inline
	// SimpleFile_Logger<MirrorBase>::SimpleFile_Logger(const std::filesystem::path& filePath, Flags initFlags, Streamable& mirrorStream)
	// 	: MirrorBase {initFlags, mirrorStream}, _file_path {filePath} {
	// 	MirrorBase::stream() << "\nSimpleFile_Logger: " << _file_path << '\n';
	// }

	// template<typename MirrorBase>
	// inline
	// Streamable& SimpleFile_Logger<MirrorBase>::stream() {
	// 	if (MirrorBase::is_cout() || !open()) {
	// 		Logger::ostreamPtr streamPtr {&_dataFile};

	// 		mirror_stream(streamPtr);
	// 		return *streamPtr;
	// 	}

	// 	return _dataFile;
	// }

	// template<typename MirrorBase>
	// inline
	// bool SimpleFile_Logger<MirrorBase>::open() {
	// 	if (!_dataFile.is_open()) {
	// 		_dataFile.open(_file_path, std::ios::app);	// Append
	// 	}

	// 	return _dataFile.good();
	// }

	// template<typename MirrorBase>
	// inline
	// Logger& SimpleFile_Logger<MirrorBase>::logTime() {
	// 	auto streamPtr {&stream()};
	// 	auto* logger {mirror_stream(streamPtr)};

	// 	while (streamPtr) {
	// 		*streamPtr << _file_path << " ";
	// 		logger = logger->mirror_stream(streamPtr);
	// 	}

	// 	MirrorBase::logTime();
	// 	return *this;
	// }

	// template<typename MirrorBase>
	// inline 
	// void SimpleFile_Logger<MirrorBase>::flush() {
	// 	auto streamPtr {&stream()};
	// 	auto* logger {mirror_stream(streamPtr)};

	// 	while (streamPtr && logger != this) {
	// 		logger->flush();
	// 		logger = logger->mirror_stream(streamPtr);
	// 	}

	// 	MirrorBase::flush();
	// 	_dataFile.flush();
	// }

	// template<typename MirrorBase>
	// inline
	// Logger* SimpleFile_Logger<MirrorBase>::mirror_stream(Logger::ostreamPtr& mirrorStream) {
	// 	bool isChainedMirror {this != _mirror};

	// 	if (isChainedMirror) {
	// 		mirrorStream = &_mirror->stream();
	// 		return _mirror;
	// 	}

	// 	return MirrorBase::mirror_stream(mirrorStream);
	// }

	// /// <summary>
	// /// Logs to file, and mirrors to the provided ostream - typcally clog
	// /// New Filenames are generated for each day
	// /// </summary>
	// template<typename MirrorBase = Console_Logger>
	// class File_Logger : public MirrorBase {
	// public:
	// 	File_Logger(const std::filesystem::path& filePath) : File_Logger {filePath, L_null} {}
	// 	File_Logger(const std::filesystem::path& filePath, Flags initFlags, Streamable& mirrorStream = std::clog);
	// 	File_Logger(const std::filesystem::path& filePath, Flags initFlags, Logger& mirror_chain) : File_Logger {filePath, initFlags} { _mirror = &mirror_chain; }

	// 	Streamable& stream() override;
	// 	void flush() override;
	// 	Logger* mirror_stream(Logger::ostreamPtr& mirrorStream) override;
	// 	bool open() override;

	// private:
	// 	Logger& logTime() override;

	// 	FileNameGenerator _fileNameGenerator;
	// 	Logger* _mirror {this};
	// 	std::ofstream _dataFile;
	// };

	// template<typename MirrorBase>
	// inline
	// File_Logger<MirrorBase>::File_Logger(const std::filesystem::path& filePath, Flags initFlags, Streamable& mirrorStream)
	// 	: MirrorBase {initFlags, mirrorStream}, _fileNameGenerator {filePath} {
	// 	MirrorBase::stream() << "\nFile_Logger: " << _fileNameGenerator.stem() << '\n';
	// }

	// template<typename MirrorBase>
	// inline
	// Streamable& File_Logger<MirrorBase>::stream() {
	// 	if (MirrorBase::is_cout() || !open()) {
	// 		Logger::ostreamPtr streamPtr {&_dataFile};

	// 		mirror_stream(streamPtr);
	// 		return *streamPtr;
	// 	}

	// 	return _dataFile;
	// }

	// template<typename MirrorBase>
	// inline
	// bool File_Logger<MirrorBase>::open() {
	// 	if (_fileNameGenerator.isNewDay(*this))
	// 		_dataFile.close();

	// 	if (!_dataFile.is_open()) {
	// 		_dataFile.open(_fileNameGenerator(*this), std::ios::app);	// Append
	// 	}

	// 	return _dataFile.good();
	// }

	// template<typename MirrorBase>
	// inline
	// Logger& File_Logger<MirrorBase>::logTime() {
	// 	auto streamPtr {&stream()};
	// 	auto* logger {mirror_stream(streamPtr)};

	// 	while (streamPtr) {
	// 		*streamPtr << _fileNameGenerator.stem() << " ";
	// 		logger = logger->mirror_stream(streamPtr);
	// 	}

	// 	MirrorBase::logTime();
	// 	return *this;
	// }

	// template<typename MirrorBase>
	// inline
	// void File_Logger<MirrorBase>::flush() {
	// 	auto streamPtr {&stream()};
	// 	auto* logger {mirror_stream(streamPtr)};

	// 	while (streamPtr && logger != this) {
	// 		logger->flush();
	// 		logger = logger->mirror_stream(streamPtr);
	// 	}

	// 	MirrorBase::flush();
	// 	_dataFile.flush();
	// }

	// template<typename MirrorBase>
	// inline
	// Logger* File_Logger<MirrorBase>::mirror_stream(Logger::ostreamPtr& mirrorStream) {
	// 	bool isChainedMirror {this != _mirror};

	// 	if (isChainedMirror) {
	// 		mirrorStream = &_mirror->stream();
	// 		return _mirror;
	// 	}

	// 	return MirrorBase::mirror_stream(mirrorStream);
	// }

	// inline
	// 	FileNameGenerator::FileNameGenerator(const std::filesystem::path& filePath) : _filePath {filePath} {
		
	// 	_fileNameStem = _filePath.filename().string();
	// 	_fileNameStem.resize(FILE_NAME_LENGTH - 4);

	// 	cerr << "FileNameGenerator::FileNameGenerator(), filePath" << filePath << ", _fileNameStem=" <<  _fileNameStem << endl;

	// 	if (!_filePath.has_extension())
	// 		_filePath += ".txt";
	// }

	// inline
	// 	std::string FileNameGenerator::operator()(const Logger& logger) {
	// 	if (logger.log_date.dayNo == 0)
	// 		logger.getTime();

	// 	_fileDayNo = logger.log_date.dayNo;

	// 	auto fileName {std::stringstream {}};

	// 	fileName << _fileNameStem << std::setfill('0') << std::setw(2) << (int)logger.log_date.monthNo << std::setw(2) << (int)_fileDayNo;
	// 	_filePath.replace_filename(fileName.str()) += _filePath.extension();
	// 	return _filePath.string();
	// }

	// class Ram_Buffer : public std::streambuf {	// derive because std::streambuf constructor is protected
	// public:
	// 	Ram_Buffer(char* start, size_t size, Logger& logger) : _logger {&logger} { setp(start, start + size); }
	// 	void empty_buffer() { setp(pbase(), epptr()); }
	// 	auto start() const { return pbase(); }
	// 	auto pos() const { return pptr(); }

	// private:
	// 	int_type overflow(int_type ch) override {
	// 		_logger->flush();
	// 		sputc(static_cast<char>(ch));
	// 		return std::char_traits<char>::not_eof(0);
	// 	}

	// 	Logger* _logger;
	// };

	// /// <summary>
	// /// Logs to RAM and flushes to file when ram-buffer is full.
	// /// Mirrors to the provided ostream - typcally cout.
	// /// </summary>
	// template<typename MirrorBase = Logger>
	// class RAM_Logger : public File_Logger<MirrorBase> {
	// public:
	// 	RAM_Logger(uint16_t ramFile_size, const std::string& fileNameStem, Flags initFlags, std::ostream& ostream = std::clog);
	// 	std::ostream& stream() override { return _stream; }
	// 	void flush() override;

	// private:
	// 	std::unique_ptr<char[]> _ramFile;
	// 	Ram_Buffer _ramBuffer;
	// 	std::ostream _stream;
	// };

	// template<typename MirrorBase>
	// 	inline
	// RAM_Logger<MirrorBase>::RAM_Logger(uint16_t ramFile_size, const std::string& fileNameStem, Flags initFlags, std::ostream& ostream)
	// 	: File_Logger<MirrorBase> {fileNameStem, initFlags, ostream}
	// 	, _ramFile {std::make_unique<char[]>(ramFile_size)}
	// 	, _ramBuffer {_ramFile.get(), ramFile_size, *this}
	// 	, _stream {&_ramBuffer} {}

	// template<typename MirrorBase>
	// 	inline
	// void RAM_Logger<MirrorBase>::flush() {
	// 	for (char* c = _ramBuffer.start(); c < _ramBuffer.pos(); ++c)
	// 		File_Logger<MirrorBase>::stream() << *c;

	// 	_ramBuffer.empty_buffer();
	// }

	inline Logger& logger() {
		static Console_Logger std_log {};

		return std_log;
	}



	inline Logger& Logger::operator <<(Flags flag) {
		if (is_null())
			return *this;

		switch (flag) {
			case L_time:
				logTime();
				break;

			case L_flush:
				_flags = static_cast<Flags>(_flags & L_allwaysFlush); // all zero's except L_allwaysFlush if set.
				*this << " |F|\n";
				flush();
				break;

			case L_endl:
			{
				if (_flags & L_allwaysFlush) {
					*this << " |F|";
				} else if (_flags == L_startWithFlushing) {
					*this << " |SF|";
				}

				auto streamPtr {&stream()};
				auto* logger {this};

				do {
					*streamPtr << "\n";
					logger = logger->mirror_stream(streamPtr);
				} while (streamPtr);

				if (_flags & L_allwaysFlush || _flags == L_startWithFlushing)
					flush();
			}

			[[fallthrough]];
			case L_clearFlags:
				if (_flags != L_startWithFlushing)
					_flags = static_cast<Flags>(_flags & L_allwaysFlush); // all zero's except L_allwaysFlush if set.

				break;

			case L_allwaysFlush:
				_flags += L_allwaysFlush;
				break;

			case L_concat:
				removeFlag(L_tabs);
				break;

			default:
				addFlag(flag);
		}

		return *this;
	}

	inline tm* Logger::getTime() {
		auto now {std::time(nullptr)};
		auto localTime {std::localtime(&now)};

		log_date.dayNo = static_cast<unsigned char>(localTime->tm_mday);
		log_date.monthNo = static_cast<unsigned char>(localTime->tm_mon + 1);
		return localTime;
	}

	inline Logger& Logger::logTime() {
		// Karl: switched to ISO 8601 type of date formatting
		*this << std::put_time(getTime(), "%Y-%m-%d %H:%M:%S");
		_flags += L_time;
		return *this;
	}
}



// int main()
// {
// 	std::cerr << "Hello World!\n";
// 	logger() << L_time << "Console_Logger is null\n";
// 	logger().activate();
// 	logger() << "Console_Logger is active\n";
// 	logger() << "Location: " << L_location << '\n';
// 	logger() << "Console_Logger hex " << std::hex << 700 << L_flush;
// 	logger() << "Console_Logger dec " << std::dec << 700 << '\n';
// 	logger() << "Console_Logger tabs " << L_tabs << std::dec << 700 << 300 << L_concat << "Done\n";
// 	logger() << "Console_Logger width " << std::setbase(16) << std::setw(10) << 10 << std::setw(5) << 19 << "Done\n";
// 	logger() << L_time << "Console_Logger time\n";
// 	logger() << L_time << "Console_Logger widget: " << widget << '\n';
// 	file1_logger() << L_flush << "StartFile1\n";
// 	file1_logger() << L_time << "new data\n";
// 	file1_logger() << L_time << "Flushed more data" << L_flush;
// 	file1_logger() << L_time << L_tabs << "yet" << "more" << "data\n";
// 	file2_logger() << L_flush << "StartFile2\n";
// 	file2_logger() << L_time << "File2 time\n";
// 	ram_logger() << L_flush << "RamFile\n";
// 	ram_logger() << L_time << "Ram data\n";
// 	ram_logger() << L_time << "Flushed Ram data" << L_flush;
// 	ram_logger() << L_time << L_tabs << "yet" << "more" << "Ram data" << L_flush;
// }

}
#endif // LOGGING_LOGGER_H