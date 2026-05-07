"use strict";
cc._RF.push(module, 'd4300SSbLBAmqdaWSSlm/Ka', 'mian');
// 3.16小游戏/command_TypeScript/level1/mian.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var main = /** @class */ (function (_super) {
    __extends(main, _super);
    function main() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.label = null;
        _this.text = 'hello';
        return _this;
    }
    main_1 = main;
    main.prototype.start = function () {
        this.node.on('touchstart', this.onEventStart, this);
    };
    main.prototype.onEventStart = function () {
        cc.log("click");
        main_1.attack = true;
    };
    var main_1;
    // LIFE-CYCLE CALLBACKS:
    // onLoad () {}
    main.attack = false;
    main.minion_attack = false;
    main.main_hp = 163;
    main.minion1_hp = 163;
    main.minion2_hp = 163;
    main.minion3_hp = 163;
    __decorate([
        property(cc.Label)
    ], main.prototype, "label", void 0);
    __decorate([
        property
    ], main.prototype, "text", void 0);
    main = main_1 = __decorate([
        ccclass
    ], main);
    return main;
}(cc.Component));
exports.default = main;

cc._RF.pop();