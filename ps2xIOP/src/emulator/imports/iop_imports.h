#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ps2x::iop::detail
{
    class IopMemory;

    struct IopImportCall
    {
        std::string library;
        uint16_t ordinal = 0;
        uint16_t version = 0;
    };

    class IopImportRegistry
    {
    public:
        explicit IopImportRegistry(IopMemory &memory) noexcept;

        void reset();
        [[nodiscard]] std::optional<IopImportCall> decode(uint32_t pc) const;
        [[nodiscard]] bool registerExportTable(uint32_t address);
        [[nodiscard]] bool releaseExportTable(uint32_t address);
        [[nodiscard]] uint32_t findTable(std::string_view library, std::optional<uint16_t> version = std::nullopt) const;
        [[nodiscard]] uint32_t resolve(std::string_view library, uint16_t ordinal, std::optional<uint16_t> version = std::nullopt) const;
        [[nodiscard]] int32_t setRebootTimeLibraryHandlingMode(uint32_t address, uint32_t mode);
        void eraseRange(uint32_t base, uint32_t size);

    private:
        struct ExportLibrary
        {
            uint32_t tableAddress = 0;
            uint16_t version = 0;
            std::string name;
            std::vector<uint32_t> functions;
        };

        [[nodiscard]] const ExportLibrary *findLibrary(std::string_view name, std::optional<uint16_t> version) const;

        std::optional<IopImportCall> decodeUncached(uint32_t pc, uint32_t delay, uint32_t &tableOut) const;

        IopMemory &m_memory;
        std::map<uint32_t, ExportLibrary> m_libraries;
        // decode() walks up to 64 KiB backwards to find a stub's import table; remember results per
        // stub (revalidated against the stub words and the table magic on every hit).
        struct DecodeCacheEntry
        {
            uint32_t delay = 0;
            uint32_t table = 0; // 0 = not an import stub
            IopImportCall call;
        };
        mutable std::unordered_map<uint32_t, DecodeCacheEntry> m_decodeCache;
    };
}
