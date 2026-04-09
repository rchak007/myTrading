#!/usr/bin/env node
/**
 * sui_swap_helper.js — Node.js helper for SUI swaps via 7K Protocol SDK
 *
 * 7K is the leading SUI DEX aggregator — routes across Aftermath, Cetus,
 * DeepBook, DeepBookV3, Turbos, FlowX, Kriya, BlueMove, and more.
 *
 * Called by Python bot.py as a subprocess:
 *   node sui_swap_helper.js \
 *     --coinIn  "0xdeeb...::deep::DEEP" \
 *     --coinOut "0xdba3...::usdc::USDC" \
 *     --amount  "180000000000" \
 *     --wallet  "0x246c..." \
 *     --privateKey "hex_ed25519_key" \
 *     --slippage 0.03
 *
 * Outputs JSON to stdout on success:
 *   {"status":"success","digest":"AbCdEf...","amountOut":"4986082991"}
 *
 * Outputs JSON to stdout on error:
 *   {"status":"error","message":"..."}
 *
 * Setup (in core/execution/):
 *   npm install @7kprotocol/sdk-ts @mysten/sui@1 --legacy-peer-deps
 */

async function main() {
    // Parse CLI args
    const args = parseArgs(process.argv.slice(2));

    const {
        coinIn,
        coinOut,
        amount,
        wallet,
        privateKey,
        slippage = "0.03",
        rpcUrl,
    } = args;

    if (!coinIn || !coinOut || !amount || !wallet || !privateKey) {
        output({ status: "error", message: "Missing required args: --coinIn --coinOut --amount --wallet --privateKey" });
        process.exit(1);
    }

    try {
        // Dynamic imports (ESM modules)
        const { SuiClient, getFullnodeUrl } = await import("@mysten/sui/client");
        const { Ed25519Keypair } = await import("@mysten/sui/keypairs/ed25519");
        const { decodeSuiPrivateKey } = await import("@mysten/sui/cryptography");
        const sevenK = await import("@7kprotocol/sdk-ts");

        // Initialize SUI client
        const suiClient = new SuiClient({
            url: rpcUrl || getFullnodeUrl("mainnet"),
        });

        // Configure 7K SDK with our SUI client (if setSuiClient exists)
        if (sevenK.default && typeof sevenK.default.setSuiClient === "function") {
            sevenK.default.setSuiClient(suiClient);
        }

        // Step 1: Get quote
        output_log(`Getting quote: ${coinIn.split("::").pop()} -> ${coinOut.split("::").pop()} amount=${amount}`);

        const getQuote = sevenK.default?.getQuote || sevenK.getQuote;
        if (!getQuote) {
            throw new Error("Could not find getQuote in @7kprotocol/sdk-ts exports: " + Object.keys(sevenK).join(", "));
        }

        const quoteResponse = await getQuote({
            tokenIn: coinIn,
            tokenOut: coinOut,
            amountIn: amount,
        });

        if (!quoteResponse) {
            throw new Error("No quote returned for this pair/amount");
        }

        const expectedOut = quoteResponse.returnAmountWithDecimal || quoteResponse.returnAmount || "?";
        output_log(`Quote received: expectedOut=${expectedOut}`);

        // Step 2: Build transaction
        output_log("Building transaction...");

        const buildTx = sevenK.default?.buildTx || sevenK.buildTx;
        if (!buildTx) {
            throw new Error("Could not find buildTx in @7kprotocol/sdk-ts exports: " + Object.keys(sevenK).join(", "));
        }

        const buildResult = await buildTx({
            quoteResponse,
            accountAddress: wallet,
            slippage: parseFloat(slippage),
            commission: {
                partner: wallet,  // Use own wallet as partner (no fee when commissionBps=0)
                commissionBps: 0,
            },
        });

        const { tx } = buildResult || {};
        if (!tx) {
            throw new Error("buildTx returned no transaction object");
        }

        output_log("Transaction built successfully");

        // Step 3: Sign and execute
        // Parse the private key
        let keypair;
        if (privateKey.startsWith("suiprivkey")) {
            const parsed = decodeSuiPrivateKey(privateKey);
            keypair = Ed25519Keypair.fromSecretKey(parsed.secretKey);
        } else {
            // Raw hex key (32 bytes)
            const keyBytes = Buffer.from(privateKey.replace("0x", ""), "hex");
            keypair = Ed25519Keypair.fromSecretKey(keyBytes);
        }

        // Sign
        tx.setSender(wallet);
        const txBytes = await tx.build({ client: suiClient });
        const { signature } = await keypair.signTransaction(txBytes);

        output_log("Transaction signed, executing...");

        // Execute
        const result = await suiClient.executeTransactionBlock({
            transactionBlock: txBytes,
            signature: [signature],
            options: {
                showEffects: true,
                showEvents: true,
            },
        });

        const digest = result.digest;
        const status = result.effects?.status?.status;

        if (status === "success") {
            output({
                status: "success",
                digest: digest,
                amountOut: String(expectedOut),
            });
        } else {
            const error = result.effects?.status?.error || "unknown";
            output({
                status: "error",
                message: `Transaction failed on-chain: ${error}`,
                digest: digest,
            });
            process.exit(1);
        }

    } catch (err) {
        output({
            status: "error",
            message: err.message || String(err),
            stack: err.stack ? err.stack.split("\n").slice(0, 3).join(" | ") : undefined,
        });
        process.exit(1);
    }
}

function parseArgs(argv) {
    const args = {};
    for (let i = 0; i < argv.length; i++) {
        if (argv[i].startsWith("--") && i + 1 < argv.length) {
            const key = argv[i].slice(2);
            args[key] = argv[i + 1];
            i++;
        }
    }
    return args;
}

function output(obj) {
    // Only JSON to stdout — Python parses this
    console.log(JSON.stringify(obj));
}

function output_log(msg) {
    // Logs to stderr so Python can capture stdout cleanly
    process.stderr.write(`[sui_swap_helper] ${msg}\n`);
}

main();